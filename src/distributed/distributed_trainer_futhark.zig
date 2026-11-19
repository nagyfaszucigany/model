const std = @import("std");
const builtin = @import("builtin");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const checkpoint_envelope = @import("checkpoint_envelope.zig");
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const gpu_memory = @import("../hw/accel/gpu_memory.zig");
const compact_batch = @import("../hw/accel/compact_batch.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray2DF16 = accel.FutharkArray2DF16;
const FutharkArray3DF16 = accel.FutharkArray3DF16;
const futhark = @import("../hw/accel/futhark_bindings.zig");
const core_relational = @import("../core_relational/mod.zig");
const CREVPipeline = core_relational.CREVPipeline;
const ChaosCoreKernel = core_relational.ChaosCoreKernel;
const nsir = core_relational.nsir_core;
const SelfSimilarRelationalGraph = core_relational.SelfSimilarRelationalGraph;
const RelationalGraphProcessingUnit = core_relational.RelationalGraphProcessingUnit;
const FNDSManager = core_relational.FNDSManager;
const PatternLocation = core_relational.PatternLocation;
const Tensor = @import("../core/tensor.zig").Tensor;
const sfd = @import("../optimizer/sfd.zig");
const mmap_dataset = @import("mmap_token_dataset.zig");
pub const CHECKPOINT_MAGIC = checkpoint_envelope.MAGIC;
pub const CHECKPOINT_VERSION = checkpoint_envelope.VERSION;
pub const CHECKPOINT_TRAILER = checkpoint_envelope.TRAILER;
pub const sfd_fisher_gamma_default: f32 = 0.99;
pub const sfd_fisher_epsilon_default: f32 = 1e-8;
pub const fused_logdet_weight_default: f32 = -1e-3;
pub const MGTLanguage = @typeInfo(@TypeOf(MGT.init)).@"fn".params[4].type.?;
pub const TokenizerFactory = *const fn (
    allocator: std.mem.Allocator,
    vocabulary: []const []const u8,
    anchors: []const []const u8,
    max_merges: usize,
) anyerror!MGT;
const OwnedTokenList = struct {
    allocator: std.mem.Allocator,
    items: [][]const u8,
    fn deinit(self: *OwnedTokenList) void {
        for (self.items) |item| self.allocator.free(item);
        self.allocator.free(self.items);
    }
};
fn appendOwnedToken(
    allocator: std.mem.Allocator,
    items: *std.ArrayList([]const u8),
    token: []const u8,
) !void {
    if (token.len == 0) return TrainerError.InvalidTokenizerData;
    const owned = try allocator.dupe(u8, token);
    items.append(owned) catch |err| {
        allocator.free(owned);
        return err;
    };
}
fn loadTokenList(
    allocator: std.mem.Allocator,
    path: []const u8,
    maximum_size: usize,
) !OwnedTokenList {
    if (path.len == 0 or maximum_size == 0) return TrainerError.InvalidTokenizerData;
    const file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{ .mode = .read_only })
    else
        try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();
    const length_u64 = try file.getEndPos();
    const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
    if (length == 0 or length > maximum_size) return TrainerError.InvalidTokenizerData;
    const data = try allocator.alloc(u8, length);
    defer allocator.free(data);
    try file.reader().readNoEof(data);
    var items = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (items.items) |item| allocator.free(item);
        items.deinit();
    }
    var content = std.mem.trimLeft(u8, data, " \t\r\n");
    if (content.len >= 3 and std.mem.eql(u8, content[0..3], "\xEF\xBB\xBF")) {
        content = std.mem.trimLeft(u8, content[3..], " \t\r\n");
    }
    if (content.len == 0) return TrainerError.InvalidTokenizerData;
    if (content[0] == '[') {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            content,
            .{ .allocate = .alloc_always },
        );
        defer parsed.deinit();
        switch (parsed.value) {
            .array => |array| {
                for (array.items) |value| {
                    switch (value) {
                        .string => |token| try appendOwnedToken(allocator, &items, token),
                        else => return TrainerError.InvalidTokenizerData,
                    }
                }
            },
            else => return TrainerError.InvalidTokenizerData,
        }
    } else {
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |raw_line| {
            const line = if (raw_line.len > 0 and raw_line[raw_line.len - 1] == '\r')
                raw_line[0 .. raw_line.len - 1]
            else
                raw_line;
            if (line.len == 0) continue;
            try appendOwnedToken(allocator, &items, line);
        }
    }
    if (items.items.len == 0) return TrainerError.InvalidTokenizerData;
    return OwnedTokenList{
        .allocator = allocator,
        .items = try items.toOwnedSlice(),
    };
}
pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.0,
    optimizer_warmup_steps: u64 = 10,
    max_line_size: usize = 10 * 1024 * 1024,
    max_tokenizer_file_size: usize = 1024 * 1024 * 1024,
    checkpoint_version: u32 = CHECKPOINT_VERSION,
    reasoning_cycles: usize = 1,
    fnds_kg_max_depth: usize = 4,
    fnds_kg_branching: usize = 3,
    knowledge_fnds_index_name: []const u8 = "knowledge_graph_patterns",
    embedding_seed: u64 = 42,
    spectral_iterations: usize = 30,
    init_spectral_iterations: usize = 0,
    spectral_target_norm: f32 = 0.9,
    gradient_clip_norm: f32 = 1.0,
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    use_normalized_gradient_flow: bool = true,
    default_max_seq_len: usize = 256,
    max_id_length: usize = 1 << 20,
    max_edge_group_count: usize = 1 << 24,
    max_node_data_length: usize = 1 << 24,
    max_node_count: usize = 1 << 20,
    max_distributed_integer: u64 = 16_777_216,
    max_local_batch_size: usize = 1 << 20,
    training_sample_count: usize = 0,
    tokenizer_vocabulary: []const []const u8 = &.{},
    tokenizer_anchors: []const []const u8 = &.{},
    tokenizer_vocabulary_path: ?[]const u8 = null,
    tokenizer_anchors_path: ?[]const u8 = null,
    tokenizer_max_merges: usize = 50_000,
    tokenizer_language: ?MGTLanguage = null,
    tokenizer_factory: ?TokenizerFactory = null,
    esso_initial_temperature: f64 = 1.0,
    esso_cooling_rate: f64 = 0.995,
    esso_max_iterations: usize = 100,
    relational_gpu_rows: usize = 4,
    relational_gpu_columns: usize = 4,
    relational_pass_interval: usize = 1000,
    reconstruction_alpha: f32 = 0.3,
    phase_a_steps: u64 = 500,
    phase_b_steps: u64 = 2000,
    max_knowledge_graph_input: usize = 64 * 1024 * 1024,
    shuffle_target_control: bool = false,
    target_source_frozen: bool = true,
    spectral_depth_compensation: bool = true,
    logdet_weight: f32 = fused_logdet_weight_default,
    fisher_gamma: f32 = sfd_fisher_gamma_default,
    fisher_epsilon: f32 = sfd_fisher_epsilon_default,
    trust_ratio: f32 = 0.1,
    weight_floor: f32 = 1e-3,
    spectral_interval: u64 = 10,
};
pub const TrainerComponents = struct {
    tokenizer: MGT,
};
const BatchBufferPool = struct {
    allocator: std.mem.Allocator,
    inputs: []u32,
    targets: []u32,
    lengths: []usize,
    fn init(allocator: std.mem.Allocator, batch_size: usize, sequence_length: usize) !BatchBufferPool {
        if (batch_size == 0 or sequence_length == 0) return TrainerError.InvalidBatchSize;
        const token_capacity = try std.math.mul(usize, batch_size, sequence_length);
        const inputs = try allocator.alloc(u32, token_capacity);
        errdefer allocator.free(inputs);
        const targets = try allocator.alloc(u32, token_capacity);
        errdefer allocator.free(targets);
        const lengths = try allocator.alloc(usize, batch_size);
        errdefer allocator.free(lengths);
        return BatchBufferPool{
            .allocator = allocator,
            .inputs = inputs,
            .targets = targets,
            .lengths = lengths,
        };
    }
    fn deinit(self: *BatchBufferPool) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
        self.allocator.free(self.lengths);
        self.* = undefined;
    }
};
pub const TrainerError = error{
    InvalidModelDim,
    InvalidNumLayers,
    InvalidBatchSize,
    InvalidWorldSize,
    InvalidRank,
    InvalidMaxLineSize,
    InvalidCheckpointVersion,
    InvalidLearningRate,
    InvalidMomentum,
    InvalidWeightsShape,
    InvalidWeightValue,
    InvalidClipRange,
    InvalidLoss,
    InvalidPinnedMemorySize,
    IndexOutOfBounds,
    TokenIndexOutOfRange,
    CheckpointVersionMismatch,
    CheckpointMagicMismatch,
    CheckpointCorrupted,
    ModelDimMismatch,
    NumLayersMismatch,
    VocabSizeMismatch,
    EmptyDataset,
    InvalidDatasetPartition,
    InvalidEnvironmentValue,
    ValueOverflow,
    ConvertPrecisionLoss,
    AllocationFailed,
    InvalidQualityByte,
    NodeIdTooLong,
    NodeDataTooLong,
    NodeCountTooLarge,
    EdgeCountTooLarge,
    DistributedIntegerPrecisionExceeded,
    InvalidDistributedInteger,
    InvalidReductionWeight,
    InvalidFloat16Value,
    InvalidGradient,
    InvalidQuantumState,
    InvalidEmbeddingWeight,
    InvalidEmbeddingShape,
    InvalidEdgeWeight,
    InvalidGraphIdentifier,
    InvalidTokenizerData,
    InvalidTokenizerConfiguration,
    InvalidOptimizerConfiguration,
    InvalidRelationalGPUConfiguration,
    InvalidSpectralState,
    InvalidCheckpointEmbeddingFlag,
    TrailingCheckpointData,
    TimestampOutOfRange,
    FileTooLarge,
    FutharkContextUnavailable,
    FutharkForwardFailed,
    FutharkTransformFailed,
    FutharkBackwardTransformFailed,
    FutharkGradientFailed,
    FutharkFullGradientFailed,
    FutharkProjectionFailed,
    FutharkGradientCopyFailed,
    EmptyKnowledgeGraphInput,
    KnowledgeGraphInputTooLarge,
    DistributedConfigMismatch,
    InvalidRelationalPassInterval,
    RelationalPassFailed,
    MemoryAdmissionRejected,
    DeviceMemoryQueryFailed,
    CheckpointSaveFailed,
    CheckpointSaveMustRunOnRoot,
    StepSynchronizerUnavailable,
    InvalidTrainingState,
    PeerRankFailure,
};
fn createConfiguredTokenizer(
    allocator: std.mem.Allocator,
    config: TrainerConfig,
) !MGT {
    if (config.tokenizer_max_merges == 0 or config.max_tokenizer_file_size == 0) return TrainerError.InvalidTokenizerConfiguration;
    var vocabulary_path_owned: ?[]u8 = null;
    defer if (vocabulary_path_owned) |path| allocator.free(path);
    var anchors_path_owned: ?[]u8 = null;
    defer if (anchors_path_owned) |path| allocator.free(path);
    var owned_vocabulary: ?OwnedTokenList = null;
    defer if (owned_vocabulary) |*list| list.deinit();
    var owned_anchors: ?OwnedTokenList = null;
    defer if (owned_anchors) |*list| list.deinit();
    var vocabulary = config.tokenizer_vocabulary;
    if (vocabulary.len == 0) {
        const vocabulary_path = config.tokenizer_vocabulary_path orelse blk: {
            vocabulary_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_VOCAB") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return TrainerError.InvalidTokenizerConfiguration,
                else => return err,
            };
            break :blk vocabulary_path_owned.?;
        };
        owned_vocabulary = try loadTokenList(allocator, vocabulary_path, config.max_tokenizer_file_size);
        vocabulary = owned_vocabulary.?.items;
    }
    if (vocabulary.len == 0) return TrainerError.InvalidTokenizerData;
    var anchors = config.tokenizer_anchors;
    if (anchors.len == 0) {
        const configured_anchor_path = config.tokenizer_anchors_path;
        const anchor_path_opt: ?[]const u8 = if (configured_anchor_path) |path|
            path
        else blk: {
            anchors_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_ANCHORS") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => break :blk null,
                else => return err,
            };
            break :blk anchors_path_owned.?;
        };
        if (anchor_path_opt) |anchor_path| {
            owned_anchors = try loadTokenList(allocator, anchor_path, config.max_tokenizer_file_size);
            anchors = owned_anchors.?.items;
        }
    }
    const max_vocab_size: u32 = std.math.cast(u32, config.tokenizer_max_merges) orelse return TrainerError.InvalidTokenizerConfiguration;
    if (config.tokenizer_factory) |tokenizer_factory| {
        return tokenizer_factory(allocator, vocabulary, anchors, config.tokenizer_max_merges);
    }
    var language_name_owned: ?[]u8 = null;
    defer if (language_name_owned) |name| allocator.free(name);
    const language = config.tokenizer_language orelse blk: {
        language_name_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_LANGUAGE") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return TrainerError.InvalidTokenizerConfiguration,
            else => return err,
        };
        break :blk std.meta.stringToEnum(MGTLanguage, language_name_owned.?) orelse return TrainerError.InvalidTokenizerConfiguration;
    };
    return MGT.init(allocator, vocabulary, anchors, max_vocab_size, language);
}
fn resolveEnvUsizeDefault(
    allocator: std.mem.Allocator,
    rank: usize,
    comptime name: []const u8,
    default_value: usize,
    upper_limit: usize,
) TrainerError!usize {
    const owned = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return default_value,
        error.OutOfMemory => return TrainerError.AllocationFailed,
        else => {
            std.debug.print("[Rank {d}] ERROR: reading env '" ++ name ++ "' failed: {}\n", .{ rank, err });
            return TrainerError.InvalidEnvironmentValue;
        },
    };
    defer allocator.free(owned);
    if (owned.len == 0) {
        std.debug.print("[Rank {d}] ERROR: " ++ name ++ " invalid: empty value\n", .{rank});
        return TrainerError.InvalidEnvironmentValue;
    }
    const parsed = std.fmt.parseInt(usize, owned, 10) catch {
        std.debug.print("[Rank {d}] ERROR: " ++ name ++ " invalid: not an integer\n", .{rank});
        return TrainerError.InvalidEnvironmentValue;
    };
    if (parsed == 0 or parsed > upper_limit) {
        std.debug.print("[Rank {d}] ERROR: " ++ name ++ " invalid: {d} outside 1..{d}\n", .{ rank, parsed, upper_limit });
        return TrainerError.InvalidEnvironmentValue;
    }
    return parsed;
}
fn resolveTemporaryDirectory(allocator: std.mem.Allocator) ![]u8 {
    const names = [_][]const u8{ "JAIDE_TMP_DIR", "TMPDIR", "TEMP", "TMP" };
    for (names) |name| {
        const value = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => continue,
            error.OutOfMemory => return TrainerError.AllocationFailed,
            else => return TrainerError.InvalidEnvironmentValue,
        };
        if (value.len == 0) {
            allocator.free(value);
            continue;
        }
        return value;
    }
    const fallback: []const u8 = if (builtin.os.tag == .windows) "." else "/tmp";
    return allocator.dupe(u8, fallback);
}
fn CrcTrackingWriter(comptime WriterType: type) type {
    return struct {
        inner: WriterType,
        crc: std.hash.Crc32,
        const Self = @This();
        pub fn write(self: *Self, bytes: []const u8) WriterType.Error!usize {
            const written = try self.inner.write(bytes);
            self.crc.update(bytes[0..written]);
            return written;
        }
        pub const Writer = std.io.Writer(*Self, WriterType.Error, write);
        pub fn writer(self: *Self) Writer {
            return .{ .context = self };
        }
    };
}
fn captureFailure(failure: *?anyerror, result: anytype) ?@typeInfo(@TypeOf(result)).error_union.payload {
    if (result) |value| {
        return value;
    } else |err| {
        if (failure.* == null) failure.* = err;
        return null;
    }
}
fn reduceFailureFlag(coordinator: *GPUCoordinator, failed: bool) !bool {
    if (coordinator.world_size <= 1) return failed;
    var flag = [1]f32{if (failed) 1.0 else 0.0};
    const byte_count = @sizeOf(f32);
    const device_values = try coordinator.allocDeviceMemory(byte_count);
    defer coordinator.freeDeviceMemory(device_values);
    try coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(flag[0..]), byte_count);
    try coordinator.allReduceFloat32Max(device_values, device_values, flag.len);
    try coordinator.copyDeviceToHost(std.mem.sliceAsBytes(flag[0..]), device_values, byte_count);
    try coordinator.synchronize();
    return flag[0] != 0.0;
}
fn admitEstimate(estimate_config: gpu_memory.EstimateConfig) TrainerError!void {
    const device_memory = gpu_memory.queryDeviceMemory() catch |err| {
        std.debug.print("[Trainer] DEVICE_MEMORY_QUERY_FAILED err={s}\n", .{@errorName(err)});
        return TrainerError.DeviceMemoryQueryFailed;
    };
    const admission = gpu_memory.admit(estimate_config, device_memory) catch |err| {
        std.debug.print("[Trainer] MEMORY_ESTIMATE_FAILED err={s}\n", .{@errorName(err)});
        return TrainerError.MemoryAdmissionRejected;
    };
    std.debug.print(
        "[Trainer] memory_admit device_free={d} device_total={d} reserve={d} budget={d} persistent={d} transient={d} peak={d} rsf={d} embedding={d} activations={d} accepted={}\n",
        .{
            admission.free_bytes,
            admission.total_bytes,
            admission.reserve_bytes,
            admission.budget_bytes,
            admission.estimate.persistent_bytes,
            admission.estimate.transient_bytes,
            admission.estimate.peak_bytes,
            admission.estimate.rsf_param_bytes,
            admission.estimate.embedding_param_bytes,
            admission.estimate.activation_bytes,
            admission.accepted,
        },
    );
    if (!admission.accepted) {
        std.debug.print(
            "[Trainer] MEMORY_ADMISSION_REJECTED largest={s} bytes={d} peak={d} budget={d}\n",
            .{ admission.largest_name, admission.largest_bytes, admission.estimate.peak_bytes, admission.budget_bytes },
        );
        return TrainerError.MemoryAdmissionRejected;
    }
}
fn splitMix64(value: u64) u64 {
    var z = value +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}
fn feistelPermute(value: u64, half: u6, key: u64) u64 {
    const mask = (@as(u64, 1) << half) - 1;
    var left = (value >> half) & mask;
    var right = value & mask;
    var round: u64 = 0;
    while (round < 4) : (round += 1) {
        const mixed = splitMix64(right ^ splitMix64(key +% round)) & mask;
        const next_right = left ^ mixed;
        left = right;
        right = next_right;
    }
    return (left << half) | right;
}
fn permuteWindowPosition(position: u64, count: u64, key: u64) u64 {
    if (count <= 1) return 0;
    const bits: u7 = 64 - @clz(count - 1);
    const half: u6 = @intCast((bits + 1) / 2);
    var value = position;
    while (true) {
        value = feistelPermute(value, half, key);
        if (value < count) return value;
    }
}
fn windowPermutationKey(seed: u64, epoch: u64) u64 {
    return splitMix64(seed ^ splitMix64(epoch +% 0xA24BAED4963EE407));
}
const StepTelemetry = struct {
    step: u64 = 0,
    loss: f32 = 0.0,
    reconstruction_loss: f32 = 0.0,
    logdet_mean: f32 = 0.0,
    source_rms: f32 = 0.0,
    finalized: bool = false,
};
const StepUpdateJob = struct {
    step: u64,
    local_fraction: f32,
    learning_rate: f32,
    momentum_beta: f32,
    fisher_gamma: f32,
    fisher_epsilon: f32,
    apply_embedding_update: bool,
    apply_spectral: bool,
    local_step_increment: u64,
    fused: accel.FusedStepResult,
};
const StepSynchronizer = struct {
    mutex: std.Thread.Mutex = .{},
    telemetry: StepTelemetry = .{},
    pending_step_increments: u64 = 0,
    fn processStep(self: *StepSynchronizer, trainer: *DistributedTrainerFuthark, job_value: StepUpdateJob) !void {
        var job = job_value;
        try self.processStepJob(trainer, &job);
    }
    fn processStepJob(self: *StepSynchronizer, trainer: *DistributedTrainerFuthark, job: *StepUpdateJob) !void {
        const ctx = &trainer.accelerator.ctx;
        defer job.fused.deinit();
        var failure: ?anyerror = null;
        _ = captureFailure(&failure, ctx.sync());
        try trainer.reduceStackGradients(&job.fused, failure);
        failure = null;
        _ = captureFailure(&failure, trainer.accelerator.applyStackGradientsSFD(
            &job.fused.stack_gradient_s,
            &job.fused.stack_gradient_t,
            job.learning_rate,
            job.momentum_beta,
            job.fisher_gamma,
            job.fisher_epsilon,
            trainer.config.trust_ratio,
            trainer.config.weight_floor,
        ));
        try trainer.updateEmbeddingFromJob(job, &failure);
        if (failure == null and job.apply_spectral) {
            const spectral_started = std.time.nanoTimestamp();
            if (captureFailure(&failure, trainer.accelerator.spectralNormalizeLayers(trainer.config.spectral_target_norm)) != null) {
                _ = captureFailure(&failure, trainer.applyEmbeddingSpectralNormalization());
            }
            if (failure == null and trainer.coordinator.isRoot()) {
                const spectral_elapsed = std.time.nanoTimestamp() - spectral_started;
                std.debug.print("[Rank 0] Step {d} spectral_ms={d}\n", .{ job.step, @divTrunc(spectral_elapsed, std.time.ns_per_ms) });
            }
        }
        const scalars = if (failure == null) captureFailure(&failure, job.fused.finalize()) else null;
        var source_rms: f32 = 0.0;
        if (failure == null) {
            if (trainer.gpu_embedding) |*embedding| {
                ctx.mutex.lock();
                defer ctx.mutex.unlock();
                if (captureFailure(&failure, embedding.sourceRootMeanSquare())) |value| source_rms = value;
            }
        }
        if (failure == null and !std.math.isFinite(source_rms)) failure = TrainerError.InvalidLoss;
        if (try trainer.agreeOnStatus(failure != null)) return failure orelse TrainerError.PeerRankFailure;
        const final_scalars = scalars.?;
        var reduced_loss = final_scalars.loss;
        var reduced_reconstruction = final_scalars.reconstruction_loss;
        var reduced_logdet = final_scalars.logdet_mean;
        if (trainer.coordinator.world_size > 1) {
            var weighted = [3]f32{
                final_scalars.loss * job.local_fraction,
                final_scalars.reconstruction_loss * job.local_fraction,
                final_scalars.logdet_mean * job.local_fraction,
            };
            try trainer.allReduceFloat32Values(weighted[0..]);
            reduced_loss = weighted[0];
            reduced_reconstruction = weighted[1];
            reduced_logdet = weighted[2];
        }
        if (!std.math.isFinite(reduced_loss) or !std.math.isFinite(reduced_reconstruction) or !std.math.isFinite(reduced_logdet)) return TrainerError.InvalidLoss;
        const step_increment: u64 = job.local_step_increment;
        self.mutex.lock();
        defer self.mutex.unlock();
        const next_pending = try std.math.add(u64, self.pending_step_increments, step_increment);
        self.telemetry = .{
            .step = job.step,
            .loss = reduced_loss,
            .reconstruction_loss = reduced_reconstruction,
            .logdet_mean = reduced_logdet,
            .source_rms = source_rms,
            .finalized = true,
        };
        self.pending_step_increments = next_pending;
    }
};
pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: *RSFAccelerator,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    local_batch_size: usize,
    global_step: u64,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,
    runtime_max_seq_len: usize,
    runtime_tokenizer_max_input_bytes: usize,
    batch_buffers: BatchBufferPool,
    gpu_embedding: ?accel.EmbeddingAccelerator,
    crev_pipeline: CREVPipeline,
    crev_kernel: *ChaosCoreKernel,
    nsir_graph: *SelfSimilarRelationalGraph,
    knowledge_nsir_graph: *SelfSimilarRelationalGraph,
    r_gpu: RelationalGraphProcessingUnit,
    fnds_manager: FNDSManager,
    spectral_normalizer: sfd.SpectralNormalizer,
    gpu_spectral_u: ?accel.FutharkArray1DF32,
    gpu_spectral_v: ?accel.FutharkArray1DF32,
    knowledge_fnds_tree_id: ?[32]u8,
    knowledge_fnds_index_id: ?[]u8,
    knowledge_graph_nonce: [32]u8,
    target_source: ?accel.FrozenEmbedding,
    shuffle_control_state: u64,
    shuffle_mutex: std.Thread.Mutex,
    relational_fast_mode: bool,
    step_synchronizer: StepSynchronizer = .{},
    nccl_mutex: std.Thread.Mutex = .{},
    last_step_telemetry: StepTelemetry = .{},
    pub const StepResult = struct {
        step: u64,
        loss: f32,
        reconstruction_loss: f32,
        source_rms: f32,
        sample_weight: f64,
        logdet_mean: f32 = 0.0,
    };
    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, 1, local_batch_size, .{});
    }
    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        var tokenizer_failure: ?anyerror = null;
        const tokenizer = captureFailure(&tokenizer_failure, createConfiguredTokenizer(allocator, config));
        return initCoordinated(allocator, coordinator, model_dim, num_layers, local_batch_size, config, tokenizer, tokenizer_failure);
    }
    pub fn initWithComponents(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
        components_in: TrainerComponents,
    ) !DistributedTrainerFuthark {
        return initCoordinated(allocator, coordinator, model_dim, num_layers, local_batch_size, config, components_in.tokenizer, null);
    }
    fn initCoordinated(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
        tokenizer: ?MGT,
        pre_failure: ?anyerror,
    ) !DistributedTrainerFuthark {
        var failure = pre_failure;
        var built: ?DistributedTrainerFuthark = null;
        if (failure == null) {
            if (tokenizer) |owned_tokenizer| {
                built = captureFailure(&failure, buildLocalTrainer(allocator, coordinator, model_dim, num_layers, local_batch_size, config, owned_tokenizer));
            } else {
                failure = TrainerError.InvalidTokenizerConfiguration;
            }
        }
        const peer_failed = reduceFailureFlag(coordinator, failure != null) catch |err| {
            if (built) |*partial| partial.deinit();
            return err;
        };
        if (peer_failed) {
            if (built) |*partial| partial.deinit();
            return failure orelse TrainerError.PeerRankFailure;
        }
        var trainer = built orelse return TrainerError.InvalidTrainingState;
        trainer.verifyConfigConsistency(trainer.vocab_size) catch |err| {
            trainer.deinit();
            return err;
        };
        return trainer;
    }
    fn buildLocalTrainer(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
        tokenizer_in: MGT,
    ) !DistributedTrainerFuthark {
        var tokenizer_transferred = false;
        errdefer if (!tokenizer_transferred) {
            var t = tokenizer_in;
            t.deinit();
        };
        if (model_dim == 0 or model_dim % 2 != 0) return TrainerError.InvalidModelDim;
        if (num_layers == 0) return TrainerError.InvalidNumLayers;
        if (local_batch_size == 0) return TrainerError.InvalidBatchSize;
        if (coordinator.world_size == 0) return TrainerError.InvalidWorldSize;
        if (coordinator.rank >= coordinator.world_size) return TrainerError.InvalidRank;
        if (config.max_line_size == 0) return TrainerError.InvalidMaxLineSize;
        if (config.checkpoint_version != CHECKPOINT_VERSION) return TrainerError.InvalidCheckpointVersion;
        if (!std.math.isFinite(config.esso_initial_temperature) or config.esso_initial_temperature <= 0.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.esso_cooling_rate) or config.esso_cooling_rate <= 0.0 or config.esso_cooling_rate > 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (config.esso_max_iterations == 0) return TrainerError.InvalidOptimizerConfiguration;
        if (config.relational_gpu_rows == 0 or config.relational_gpu_columns == 0) return TrainerError.InvalidRelationalGPUConfiguration;
        if (config.knowledge_fnds_index_name.len == 0) return TrainerError.InvalidGraphIdentifier;
        if (!std.math.isFinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0.0) return TrainerError.InvalidGradient;
        if (!std.math.isFinite(config.spectral_target_norm) or config.spectral_target_norm <= 0.0) return TrainerError.InvalidSpectralState;
        if (config.spectral_interval == 0) return TrainerError.InvalidSpectralState;
        if (!std.math.isFinite(config.trust_ratio) or config.trust_ratio <= 0.0 or config.trust_ratio > 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.weight_floor) or config.weight_floor <= 0.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.fisher_gamma) or config.fisher_gamma < 0.0 or config.fisher_gamma >= 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.fisher_epsilon) or config.fisher_epsilon < 1e-12) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.clip_min) or !std.math.isFinite(config.clip_max) or config.clip_min >= config.clip_max) return TrainerError.InvalidClipRange;
        if (config.relational_pass_interval == 0) return TrainerError.InvalidRelationalPassInterval;
        if (config.default_max_seq_len == 0 or config.default_max_seq_len > config.max_distributed_integer) return TrainerError.InvalidEnvironmentValue;
        try validateHyperparameters(config.learning_rate, config.momentum);
        const runtime_max_seq_len = try resolveEnvUsizeDefault(allocator, coordinator.rank, "JAIDE_MAX_SEQ_LEN", config.default_max_seq_len, config.max_distributed_integer);
        const runtime_tokenizer_max_input_bytes = try resolveEnvUsizeDefault(allocator, coordinator.rank, "JAIDE_TOKENIZER_MAX_INPUT_BYTES", 1024, config.max_distributed_integer);
        var batch_buffers = try BatchBufferPool.init(allocator, local_batch_size, runtime_max_seq_len);
        var batch_buffers_committed = false;
        errdefer if (!batch_buffers_committed) batch_buffers.deinit();
        const actual_model_dim = model_dim;
        const vocab_size_init = tokenizer_in.next_token_id;
        try admitEstimate(gpu_memory.EstimateConfig{
            .model_dim = actual_model_dim,
            .num_layers = num_layers,
            .vocab_size = vocab_size_init,
            .batch_size = local_batch_size,
            .max_seq_len = runtime_max_seq_len,
            .world_size = coordinator.world_size,
            .stack_only = true,
            .frozen_target_f16_only = config.target_source_frozen,
            .include_graph_chunk = false,
            .spectral_temps = false,
            .coupling = .diagonal,
        });
        const accelerator_ptr = try allocator.create(RSFAccelerator);
        var accelerator_ptr_committed = false;
        errdefer if (!accelerator_ptr_committed) allocator.destroy(accelerator_ptr);
        std.debug.print(
            "[Trainer] allocating RSF stacks dim={d} layers={d} vocab={d}\n",
            .{ actual_model_dim, num_layers, tokenizer_in.next_token_id },
        );
        accelerator_ptr.* = try RSFAccelerator.initMultiLayerWithDepthScale(
            actual_model_dim,
            num_layers,
            allocator,
            config.spectral_depth_compensation,
        );
        var accelerator_committed = false;
        errdefer if (!accelerator_committed) accelerator_ptr.deinit();
        try accelerator_ptr.setClipRange(
            try checkedF32ToF16(config.clip_min),
            try checkedF32ToF16(config.clip_max),
        );
        if (config.init_spectral_iterations > 0) {
            std.debug.print("[Trainer] init spectral normalize iters={d}\n", .{config.init_spectral_iterations});
            try accelerator_ptr.spectralNormalizeLayers(config.spectral_target_norm);
        }
        std.debug.print(
            "[Trainer] allocating embeddings vocab={d} dim={d}\n",
            .{ tokenizer_in.next_token_id, actual_model_dim },
        );
        var gpu_embedding = try accel.EmbeddingAccelerator.init(
            allocator,
            &accelerator_ptr.ctx,
            tokenizer_in.next_token_id,
            actual_model_dim,
            config.embedding_seed,
        );
        var gpu_embedding_committed = false;
        errdefer if (!gpu_embedding_committed) gpu_embedding.deinit();
        var target_source: ?accel.FrozenEmbedding = null;
        var target_source_committed = false;
        errdefer if (!target_source_committed) {
            if (target_source) |*source| source.deinit();
        };
        if (config.target_source_frozen) {
            std.debug.print("[Trainer] allocating frozen target embeddings f16-only\n", .{});
            target_source = try accel.FrozenEmbedding.initFromTrainableMaster(&gpu_embedding);
        }
        std.debug.print("[Trainer] model buffers ready\n", .{});
        const crev_kernel_ptr = try allocator.create(ChaosCoreKernel);
        var crev_kernel_ptr_committed = false;
        errdefer if (!crev_kernel_ptr_committed) allocator.destroy(crev_kernel_ptr);
        crev_kernel_ptr.* = ChaosCoreKernel.init(allocator);
        var crev_kernel_committed = false;
        errdefer if (!crev_kernel_committed) crev_kernel_ptr.deinit();
        var crev_pipeline = try CREVPipeline.init(allocator, crev_kernel_ptr);
        var crev_pipeline_committed = false;
        errdefer if (!crev_pipeline_committed) crev_pipeline.deinit();
        const nsir_graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var nsir_graph_ptr_committed = false;
        errdefer if (!nsir_graph_ptr_committed) allocator.destroy(nsir_graph_ptr);
        nsir_graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var nsir_graph_committed = false;
        errdefer if (!nsir_graph_committed) nsir_graph_ptr.deinit();
        const knowledge_nsir_graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var knowledge_nsir_graph_ptr_committed = false;
        errdefer if (!knowledge_nsir_graph_ptr_committed) allocator.destroy(knowledge_nsir_graph_ptr);
        knowledge_nsir_graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var knowledge_nsir_graph_committed = false;
        errdefer if (!knowledge_nsir_graph_committed) knowledge_nsir_graph_ptr.deinit();
        var r_gpu_inst = try RelationalGraphProcessingUnit.init(
            allocator,
            config.relational_gpu_rows,
            config.relational_gpu_columns,
        );
        var r_gpu_committed = false;
        errdefer if (!r_gpu_committed) r_gpu_inst.deinit();
        var fnds_manager_inst = try FNDSManager.init(allocator);
        var fnds_manager_committed = false;
        errdefer if (!fnds_manager_committed) fnds_manager_inst.deinit();
        const spectral_normalizer = sfd.SpectralNormalizer.initWithConfig(.{
            .max_singular_value = config.spectral_target_norm,
        });
        var knowledge_graph_nonce: [32]u8 = undefined;
        var nonce_prng = std.Random.DefaultPrng.init(config.embedding_seed ^ @as(u64, @intCast(model_dim)) ^ (@as(u64, @intCast(num_layers)) << 32));
        nonce_prng.random().bytes(knowledge_graph_nonce[0..]);
        tokenizer_transferred = true;
        accelerator_ptr_committed = true;
        accelerator_committed = true;
        gpu_embedding_committed = true;
        crev_kernel_ptr_committed = true;
        crev_kernel_committed = true;
        crev_pipeline_committed = true;
        nsir_graph_ptr_committed = true;
        nsir_graph_committed = true;
        knowledge_nsir_graph_ptr_committed = true;
        knowledge_nsir_graph_committed = true;
        r_gpu_committed = true;
        fnds_manager_committed = true;
        target_source_committed = true;
        batch_buffers_committed = true;
        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = tokenizer_in,
            .accelerator = accelerator_ptr,
            .model_dim = actual_model_dim,
            .num_layers = num_layers,
            .vocab_size = tokenizer_in.next_token_id,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
            .runtime_max_seq_len = runtime_max_seq_len,
            .runtime_tokenizer_max_input_bytes = runtime_tokenizer_max_input_bytes,
            .batch_buffers = batch_buffers,
            .gpu_embedding = gpu_embedding,
            .crev_pipeline = crev_pipeline,
            .crev_kernel = crev_kernel_ptr,
            .nsir_graph = nsir_graph_ptr,
            .knowledge_nsir_graph = knowledge_nsir_graph_ptr,
            .r_gpu = r_gpu_inst,
            .fnds_manager = fnds_manager_inst,
            .spectral_normalizer = spectral_normalizer,
            .gpu_spectral_u = null,
            .gpu_spectral_v = null,
            .knowledge_fnds_tree_id = null,
            .knowledge_fnds_index_id = null,
            .knowledge_graph_nonce = knowledge_graph_nonce,
            .target_source = target_source,
            .shuffle_control_state = config.embedding_seed ^ 0x5DEECE66D,
            .shuffle_mutex = .{},
            .relational_fast_mode = if (std.posix.getenv("JAIDE_RELATIONAL_FAST")) |v| std.mem.eql(u8, v, "1") else true,
        };
    }
    fn verifyConfigConsistency(self: *DistributedTrainerFuthark, local_vocab_size: usize) !void {
        if (self.coordinator.world_size <= 1) return;
        const config_values = [_]u64{
            @as(u32, @bitCast(self.learning_rate)),
            @as(u32, @bitCast(self.momentum)),
            self.config.optimizer_warmup_steps,
            self.local_batch_size,
            self.config.reasoning_cycles,
            self.config.embedding_seed,
            self.config.spectral_iterations,
            self.config.init_spectral_iterations,
            @as(u32, @bitCast(self.config.spectral_target_norm)),
            @as(u32, @bitCast(self.config.gradient_clip_norm)),
            @as(u32, @bitCast(self.config.clip_min)),
            @as(u32, @bitCast(self.config.clip_max)),
            @intFromBool(self.config.grad_mean),
            @intFromBool(self.config.use_normalized_gradient_flow),
            self.config.default_max_seq_len,
            @as(u64, @bitCast(self.config.esso_initial_temperature)),
            @as(u64, @bitCast(self.config.esso_cooling_rate)),
            self.config.esso_max_iterations,
            self.config.relational_gpu_rows,
            self.config.relational_gpu_columns,
            self.config.relational_pass_interval,
            @as(u32, @bitCast(self.config.reconstruction_alpha)),
            self.config.phase_a_steps,
            self.config.phase_b_steps,
            @intFromBool(self.config.shuffle_target_control),
            @intFromBool(self.config.target_source_frozen),
            @intFromBool(self.config.spectral_depth_compensation),
            @as(u32, @bitCast(self.config.logdet_weight)),
            @as(u32, @bitCast(self.config.fisher_gamma)),
            @as(u32, @bitCast(self.config.fisher_epsilon)),
            @as(u32, @bitCast(self.config.trust_ratio)),
            @as(u32, @bitCast(self.config.weight_floor)),
            self.config.spectral_interval,
            self.config.max_distributed_integer,
            @intFromBool(self.relational_fast_mode),
            self.runtime_max_seq_len,
            self.runtime_tokenizer_max_input_bytes,
        };
        try self.verifyDistributedValues(config_values[0..]);
        const limit = self.config.max_distributed_integer;
        const vocab_u64: u64 = @intCast(local_vocab_size);
        const dim_u64: u64 = @intCast(self.model_dim);
        const layers_u64: u64 = @intCast(self.num_layers);
        const local_exceeds: u64 = if (vocab_u64 > limit or dim_u64 > limit or layers_u64 > limit) 1 else 0;
        const global_exceeds = try self.allReduceMaximumU64Raw(local_exceeds, 1);
        if (global_exceeds != 0) return TrainerError.DistributedIntegerPrecisionExceeded;
        const max_vocab = try self.allReduceMaximumU64Raw(vocab_u64, limit);
        const min_vocab_enc = try self.allReduceMaximumU64Raw(limit - vocab_u64, limit);
        const max_dim = try self.allReduceMaximumU64Raw(dim_u64, limit);
        const min_dim_enc = try self.allReduceMaximumU64Raw(limit - dim_u64, limit);
        const max_layers = try self.allReduceMaximumU64Raw(layers_u64, limit);
        const min_layers_enc = try self.allReduceMaximumU64Raw(limit - layers_u64, limit);
        if (max_vocab != vocab_u64 or min_vocab_enc != limit - vocab_u64) return TrainerError.DistributedConfigMismatch;
        if (max_dim != dim_u64 or min_dim_enc != limit - dim_u64) return TrainerError.DistributedConfigMismatch;
        if (max_layers != layers_u64 or min_layers_enc != limit - layers_u64) return TrainerError.DistributedConfigMismatch;
    }
    fn resetSpectralState(self: *DistributedTrainerFuthark) void {
        const ctx = &self.accelerator.ctx;
        if (self.gpu_spectral_u) |*u| {
            u.free(ctx) catch |err| std.log.err("distributed trainer: releasing spectral u vector failed: {s}", .{@errorName(err)});
        }
        if (self.gpu_spectral_v) |*v| {
            v.free(ctx) catch |err| std.log.err("distributed trainer: releasing spectral v vector failed: {s}", .{@errorName(err)});
        }
        self.gpu_spectral_u = null;
        self.gpu_spectral_v = null;
    }
    fn absorbStepTelemetry(self: *DistributedTrainerFuthark) !void {
        const synchronizer = &self.step_synchronizer;
        synchronizer.mutex.lock();
        defer synchronizer.mutex.unlock();
        const next_step = try std.math.add(u64, self.global_step, synchronizer.pending_step_increments);
        self.global_step = next_step;
        synchronizer.pending_step_increments = 0;
        self.last_step_telemetry = synchronizer.telemetry;
    }
    fn absorbStepState(self: *DistributedTrainerFuthark) !void {
        try self.absorbStepTelemetry();
    }
    fn releaseKnowledgeFndsResources(self: *DistributedTrainerFuthark) void {
        if (self.knowledge_fnds_index_id) |index_id| {
            _ = self.fnds_manager.removeIndex(index_id);
            self.allocator.free(index_id);
            self.knowledge_fnds_index_id = null;
        }
        if (self.knowledge_fnds_tree_id) |tree_id| {
            _ = self.fnds_manager.removeTree(tree_id);
            self.knowledge_fnds_tree_id = null;
        }
    }
    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch |err| {
            std.debug.print("[Rank {d}] WARN: accelerator.sync during deinit failed: {}\n", .{ self.coordinator.rank, err });
        };
        self.resetSpectralState();
        self.releaseKnowledgeFndsResources();
        self.fnds_manager.deinit();
        self.r_gpu.deinit();
        self.knowledge_nsir_graph.deinit();
        self.allocator.destroy(self.knowledge_nsir_graph);
        self.nsir_graph.deinit();
        self.allocator.destroy(self.nsir_graph);
        self.crev_pipeline.deinit();
        self.crev_kernel.deinit();
        self.allocator.destroy(self.crev_kernel);
        if (self.target_source) |*source| source.deinit();
        if (self.gpu_embedding) |*emb| emb.deinit();
        self.accelerator.deinit();
        self.allocator.destroy(self.accelerator);
        self.batch_buffers.deinit();
        self.tokenizer.deinit();
    }
    fn validateHyperparameters(learning_rate: f32, momentum: f32) TrainerError!void {
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0.0) return TrainerError.InvalidLearningRate;
        if (!std.math.isFinite(momentum) or momentum < 0.0 or momentum >= 1.0) return TrainerError.InvalidMomentum;
    }
    fn checkedF32ToF16(value: f32) TrainerError!f16 {
        if (!std.math.isFinite(value)) return TrainerError.InvalidFloat16Value;
        if (value < -65504.0 or value > 65504.0) return TrainerError.InvalidFloat16Value;
        const converted: f16 = @floatCast(value);
        if (!std.math.isFinite(converted)) return TrainerError.InvalidFloat16Value;
        return converted;
    }
    fn safeUsizeToU32(value: usize) TrainerError!u32 {
        if (value > std.math.maxInt(u32)) return TrainerError.ValueOverflow;
        return @as(u32, @intCast(value));
    }
    fn openReadFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        return std.fs.cwd().openFile(path, .{ .mode = .read_only });
    }
    fn createWriteFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .mode = 0o600, .truncate = true, .exclusive = true });
        return std.fs.cwd().createFile(path, .{ .mode = 0o600, .truncate = true, .exclusive = true });
    }
    fn deletePath(path: []const u8) void {
        if (std.fs.path.isAbsolute(path)) {
            std.fs.deleteFileAbsolute(path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => std.debug.print("WARN: deletePath({s}) failed: {}\n", .{ path, err }),
            };
            return;
        }
        std.fs.cwd().deleteFile(path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => std.debug.print("WARN: deletePath({s}) failed: {}\n", .{ path, err }),
        };
    }
    fn renamePath(from: []const u8, to: []const u8) !void {
        if (std.fs.path.isAbsolute(from)) return std.fs.renameAbsolute(from, to);
        return std.fs.cwd().rename(from, to);
    }
    fn syncContainingDirectory(path: []const u8) void {
        const dir_path = std.fs.path.dirname(path) orelse ".";
        var dir = if (std.fs.path.isAbsolute(dir_path))
            std.fs.openDirAbsolute(dir_path, .{}) catch return
        else
            std.fs.cwd().openDir(dir_path, .{}) catch return;
        defer dir.close();
        std.posix.fsync(dir.fd) catch {};
    }
    fn writeF32Array(writer: anytype, values: []const f32, require_nonnegative: bool) !void {
        try writer.writeInt(u64, @intCast(values.len), .little);
        var bytes: [16 * 1024]u8 = undefined;
        var offset: usize = 0;
        while (offset < values.len) {
            const count = @min(bytes.len / @sizeOf(f32), values.len - offset);
            var index: usize = 0;
            while (index < count) : (index += 1) {
                const value = values[offset + index];
                if (!std.math.isFinite(value) or (require_nonnegative and value < 0.0)) return TrainerError.InvalidWeightValue;
                std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little);
            }
            try writer.writeAll(bytes[0 .. count * 4]);
            offset += count;
        }
    }
    fn writeF64(writer: anytype, value: f64) !void {
        try writer.writeInt(u64, @as(u64, @bitCast(value)), .little);
    }
    fn readF64(reader: anytype) !f64 {
        const bits = try reader.readInt(u64, .little);
        return @as(f64, @bitCast(bits));
    }
    fn truncateUtf8Input(text: []const u8, maximum_bytes: usize) []const u8 {
        const length = @min(text.len, maximum_bytes);
        var boundary = length;
        while (boundary > 0 and boundary < text.len and (text[boundary] & 0xC0) == 0x80) {
            boundary -= 1;
        }
        return text[0..boundary];
    }
    fn agreeOnStatus(self: *DistributedTrainerFuthark, local_failed: bool) !bool {
        if (self.coordinator.world_size <= 1) return local_failed;
        self.nccl_mutex.lock();
        defer self.nccl_mutex.unlock();
        return reduceFailureFlag(self.coordinator, local_failed);
    }
    fn allReduceFloat32Values(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) return;
        self.nccl_mutex.lock();
        defer self.nccl_mutex.unlock();
        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32(device_values, device_values, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), device_values, byte_count);
        try self.coordinator.synchronize();
    }
    fn allReduceFloat32MaximumValues(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) return;
        self.nccl_mutex.lock();
        defer self.nccl_mutex.unlock();
        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32Max(device_values, device_values, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), device_values, byte_count);
        try self.coordinator.synchronize();
    }
    fn verifyDistributedValues(self: *DistributedTrainerFuthark, values: []const u64) !void {
        if (self.coordinator.world_size <= 1 or values.len == 0) return;
        const limbs_per_value: usize = 4;
        const entries_per_limb: usize = 2;
        const entry_count = try std.math.mul(usize, values.len, limbs_per_value * entries_per_limb);
        const entries = try self.allocator.alloc(f32, entry_count);
        defer self.allocator.free(entries);
        for (values, 0..) |value, value_index| {
            var limb_index: usize = 0;
            while (limb_index < limbs_per_value) : (limb_index += 1) {
                const shift: u6 = @intCast(limb_index * 16);
                const limb: u16 = @truncate(value >> shift);
                const offset = value_index * limbs_per_value * entries_per_limb + limb_index * entries_per_limb;
                entries[offset] = @floatFromInt(limb);
                entries[offset + 1] = @floatFromInt(std.math.maxInt(u16) - limb);
            }
        }
        try self.allReduceFloat32MaximumValues(entries);
        for (values, 0..) |value, value_index| {
            var limb_index: usize = 0;
            while (limb_index < limbs_per_value) : (limb_index += 1) {
                const shift: u6 = @intCast(limb_index * 16);
                const limb: u16 = @truncate(value >> shift);
                const offset = value_index * limbs_per_value * entries_per_limb + limb_index * entries_per_limb;
                if (entries[offset] != @as(f32, @floatFromInt(limb)) or entries[offset + 1] != @as(f32, @floatFromInt(std.math.maxInt(u16) - limb))) return TrainerError.DistributedConfigMismatch;
            }
        }
    }
    fn allReduceMaximumU64Raw(self: *DistributedTrainerFuthark, value: u64, limit: u64) !u64 {
        if (self.coordinator.world_size <= 1) return value;
        self.nccl_mutex.lock();
        defer self.nccl_mutex.unlock();
        var arr = [1]f32{@as(f32, @floatFromInt(value))};
        const byte_count = @sizeOf(f32);
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(arr[0..]), byte_count);
        try self.coordinator.allReduceFloat32Max(device_values, device_values, arr.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(arr[0..]), device_values, byte_count);
        try self.coordinator.synchronize();
        if (!std.math.isFinite(arr[0]) or arr[0] < 0.0 or arr[0] > @as(f32, @floatFromInt(limit))) return TrainerError.InvalidDistributedInteger;
        return @as(u64, @intFromFloat(arr[0]));
    }
    fn allReduceMaximumU64(self: *DistributedTrainerFuthark, value: u64) !u64 {
        if (self.coordinator.world_size <= 1) {
            if (value > self.config.max_distributed_integer) return TrainerError.DistributedIntegerPrecisionExceeded;
            return value;
        }
        const overflow_flag: u64 = if (value > self.config.max_distributed_integer) 1 else 0;
        const global_overflow = try self.allReduceMaximumU64Raw(overflow_flag, 1);
        if (global_overflow != 0) return TrainerError.DistributedIntegerPrecisionExceeded;
        return self.allReduceMaximumU64Raw(value, self.config.max_distributed_integer);
    }
    fn allReduceSumU64(self: *DistributedTrainerFuthark, value: u64) !u64 {
        if (self.coordinator.world_size <= 1) return value;
        const exact_integer_limit: u64 = 1 << 24;
        const world_size_u64 = std.math.cast(u64, self.coordinator.world_size) orelse return TrainerError.ValueOverflow;
        if (world_size_u64 == 0) return TrainerError.InvalidWorldSize;
        if (world_size_u64 > exact_integer_limit) return TrainerError.DistributedIntegerPrecisionExceeded;
        var radix_bits: u6 = 1;
        while (radix_bits < 24) {
            const candidate_bits: u6 = radix_bits + 1;
            const candidate_mask = (@as(u64, 1) << candidate_bits) - 1;
            if (candidate_mask > exact_integer_limit / world_size_u64) break;
            radix_bits = candidate_bits;
        }
        const bits_per_limb: usize = @intCast(radix_bits);
        const limb_count = (64 + bits_per_limb - 1) / bits_per_limb;
        const limb_mask = (@as(u64, 1) << radix_bits) - 1;
        var limb_values = [_]f32{0.0} ** 64;
        var limb_index: usize = 0;
        while (limb_index < limb_count) : (limb_index += 1) {
            const shift: u6 = @intCast(limb_index * bits_per_limb);
            const limb = (value >> shift) & limb_mask;
            limb_values[limb_index] = @as(f32, @floatFromInt(limb));
        }
        try self.allReduceFloat32Values(limb_values[0..limb_count]);
        var result: u128 = 0;
        var carry: u128 = 0;
        limb_index = 0;
        while (limb_index < limb_count) : (limb_index += 1) {
            const reduced = limb_values[limb_index];
            if (!std.math.isFinite(reduced) or reduced < 0.0 or reduced > @as(f32, @floatFromInt(exact_integer_limit))) return TrainerError.InvalidDistributedInteger;
            const rounded = @round(reduced);
            if (rounded != reduced) return TrainerError.InvalidDistributedInteger;
            const limb_sum: u64 = @intFromFloat(rounded);
            const total = @as(u128, limb_sum) + carry;
            const digit = total & @as(u128, limb_mask);
            const shift: u7 = @intCast(limb_index * bits_per_limb);
            result |= digit << shift;
            carry = total >> radix_bits;
        }
        if (carry != 0 or result > @as(u128, std.math.maxInt(u64))) return TrainerError.ValueOverflow;
        return @intCast(result);
    }
    fn settlePreparedBatch(self: *DistributedTrainerFuthark, failure: ?anyerror, prepared: *?PreparedBatch) !void {
        const peer_failed = self.agreeOnStatus(failure != null) catch |err| {
            if (prepared.*) |*value| value.deinit();
            prepared.* = null;
            return err;
        };
        if (peer_failed) {
            if (prepared.*) |*value| value.deinit();
            prepared.* = null;
            return failure orelse TrainerError.PeerRankFailure;
        }
    }
    fn prepareBatchAgreed(self: *DistributedTrainerFuthark, batch: [][]const u8) !PreparedBatch {
        var failure: ?anyerror = null;
        var prepared = captureFailure(&failure, self.prepareBatch(batch));
        try self.settlePreparedBatch(failure, &prepared);
        return prepared orelse TrainerError.InvalidTrainingState;
    }
    fn prepareNextBatchAgreed(self: *DistributedTrainerFuthark, next_batch: ?[][]const u8, next_prepared: *?PreparedBatch) !void {
        next_prepared.* = null;
        var failure: ?anyerror = null;
        if (next_batch) |batch| {
            next_prepared.* = captureFailure(&failure, self.prepareBatch(batch));
        }
        try self.settlePreparedBatch(failure, next_prepared);
    }
    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (self.local_batch_size == 0) return TrainerError.InvalidBatchSize;
        const local_batch_count: u64 = if (samples.len == 0) 0 else @as(u64, (samples.len - 1) / self.local_batch_size + 1);
        const target_batch_count = try self.allReduceMaximumU64(local_batch_count);
        var total_weighted_loss: f64 = 0.0;
        var total_sample_weight: f64 = 0.0;
        var batch_start: usize = 0;
        var current_prepared: ?PreparedBatch = null;
        defer if (current_prepared) |*prepared| prepared.deinit();
        if (target_batch_count > 0) {
            var first_batch: [][]const u8 = &.{};
            if (batch_start < samples.len) {
                const remaining = samples.len - batch_start;
                const batch_length = @min(self.local_batch_size, remaining);
                const batch_end = try std.math.add(usize, batch_start, batch_length);
                first_batch = samples[batch_start..batch_end];
                batch_start = batch_end;
            }
            current_prepared = try self.prepareBatchAgreed(first_batch);
        }
        var batch_index: u64 = 0;
        while (batch_index < target_batch_count) : (batch_index += 1) {
            var next_batch: ?[][]const u8 = null;
            if (batch_index + 1 < target_batch_count) {
                var batch: [][]const u8 = &.{};
                if (batch_start < samples.len) {
                    const remaining = samples.len - batch_start;
                    const batch_length = @min(self.local_batch_size, remaining);
                    const batch_end = try std.math.add(usize, batch_start, batch_length);
                    batch = samples[batch_start..batch_end];
                    batch_start = batch_end;
                }
                next_batch = batch;
            }
            var prepared_next: ?PreparedBatch = null;
            const step_result = self.trainPreparedStepFuthark(&current_prepared.?, next_batch, &prepared_next) catch |err| {
                if (prepared_next) |*prepared| prepared.deinit();
                std.debug.print("[Rank {d}] trainStepFuthark ERROR at step {d}: {}\n", .{ self.coordinator.rank, self.global_step, err });
                return err;
            };
            current_prepared.?.deinit();
            current_prepared = null;
            if (!std.math.isFinite(step_result.loss)) {
                if (prepared_next) |*prepared| prepared.deinit();
                return TrainerError.InvalidLoss;
            }
            total_weighted_loss += @as(f64, step_result.loss) * step_result.sample_weight;
            total_sample_weight += step_result.sample_weight;
            if (self.coordinator.isRoot() and (self.global_step <= 50 or self.global_step % 10 == 0)) {
                std.debug.print("[Step {d}] Loss: {d:.6} | Recon: {d:.6} | LogDet: {d:.6} | SourceRMS: {d:.6}\n", .{
                    step_result.step,
                    step_result.loss,
                    step_result.reconstruction_loss,
                    step_result.logdet_mean,
                    step_result.source_rms,
                });
            }
            if (batch_index + 1 < target_batch_count) {
                current_prepared = prepared_next orelse return TrainerError.AllocationFailed;
            } else if (prepared_next) |*prepared| {
                prepared.deinit();
            }
        }
        try self.absorbStepState();
        const reduce_buf = [2]f64{ total_weighted_loss, total_sample_weight };
        var reduce_f32 = [2]f32{
            @as(f32, @floatCast(reduce_buf[0])),
            @as(f32, @floatCast(reduce_buf[1])),
        };
        try self.allReduceFloat32Values(reduce_f32[0..]);
        const global_loss_sum: f64 = @as(f64, reduce_f32[0]);
        const global_weight: f64 = @as(f64, reduce_f32[1]);
        if (global_weight <= 0.0) {
            std.debug.print("[WARNING] No samples processed across all ranks\n", .{});
            return 0.0;
        }
        const result: f32 = @floatCast(global_loss_sum / global_weight);
        if (!std.math.isFinite(result)) return TrainerError.InvalidLoss;
        return result;
    }
    fn ensureKnowledgeFndsTree(self: *DistributedTrainerFuthark) ![32]u8 {
        if (self.knowledge_fnds_tree_id) |tree_id| return tree_id;
        const tree_id = try self.fnds_manager.createTree(self.config.fnds_kg_max_depth, self.config.fnds_kg_branching);
        self.knowledge_fnds_tree_id = tree_id;
        return tree_id;
    }
    fn ensureKnowledgeFndsIndex(self: *DistributedTrainerFuthark) ![]const u8 {
        if (self.knowledge_fnds_index_id) |index_id| return index_id;
        const index_id = try self.allocator.dupe(u8, self.config.knowledge_fnds_index_name);
        errdefer self.allocator.free(index_id);
        try self.fnds_manager.createIndex(index_id);
        if (self.fnds_manager.getIndex(index_id) == null) {
            _ = self.fnds_manager.removeIndex(index_id);
            return TrainerError.InvalidGraphIdentifier;
        }
        self.knowledge_fnds_index_id = index_id;
        return index_id;
    }
    const PreparedBatch = struct {
        allocator: std.mem.Allocator,
        token_lists: std.ArrayList(std.ArrayList(u32)),
        active_lists: std.ArrayList(std.ArrayList(u32)),
        real_sequence_lengths: []usize,
        flat_input_tokens: []u32,
        flat_target_tokens: []u32,
        effective_batch_size: usize,
        sequence_length: usize,
        local_active_samples: u64,
        local_token_count: u64,
        pooled: bool = false,
        fn deinit(self: *PreparedBatch) void {
            if (self.pooled) {
                self.* = undefined;
                return;
            }
            self.allocator.free(self.flat_target_tokens);
            self.allocator.free(self.flat_input_tokens);
            self.allocator.free(self.real_sequence_lengths);
            self.active_lists.deinit();
            for (self.token_lists.items) |*list| list.deinit();
            self.token_lists.deinit();
            self.* = undefined;
        }
    };
    const TokenEpochLayout = struct {
        window_size: usize,
        selected_windows: usize,
        rank_start: usize,
        rank_count: usize,
        local_batch_count: u64,
        global_batch_count: u64,
    };
    fn tokenEpochLayout(
        self: *DistributedTrainerFuthark,
        dataset: *const mmap_dataset.MmapTokenDataset,
        epoch: u64,
        first_batch: u64,
    ) !TokenEpochLayout {
        const window_size = std.math.add(usize, self.runtime_max_seq_len, 1) catch return TrainerError.ValueOverflow;
        const total_windows = dataset.payload.len / window_size;
        const selected_windows = if (self.config.training_sample_count == 0)
            total_windows
        else
            @min(total_windows, self.config.training_sample_count);
        const rank_bounds = try rankWindowBounds(selected_windows, self.coordinator.world_size, self.coordinator.rank);
        const local_batch_count: u64 = if (rank_bounds.count == 0) 0 else @as(u64, (rank_bounds.count - 1) / self.local_batch_size + 1);
        const global_batch_count = try self.allReduceMaximumU64(local_batch_count);
        const agreement = [_]u64{ epoch, first_batch, @intCast(selected_windows), @intCast(window_size) };
        try self.verifyDistributedValues(agreement[0..]);
        if (first_batch > global_batch_count) return TrainerError.IndexOutOfBounds;
        return TokenEpochLayout{
            .window_size = window_size,
            .selected_windows = selected_windows,
            .rank_start = rank_bounds.start,
            .rank_count = rank_bounds.count,
            .local_batch_count = local_batch_count,
            .global_batch_count = global_batch_count,
        };
    }
    fn prepareTokenBatch(
        self: *DistributedTrainerFuthark,
        payload: []const u32,
        layout: TokenEpochLayout,
        batch_index: u64,
        permutation_key: u64,
    ) !PreparedBatch {
        const batch_offset = std.math.mul(usize, @intCast(batch_index), self.local_batch_size) catch return TrainerError.ValueOverflow;
        if (batch_offset >= layout.rank_count) return TrainerError.IndexOutOfBounds;
        const batch_size = @min(self.local_batch_size, layout.rank_count - batch_offset);
        const sequence_length = self.runtime_max_seq_len;
        if (batch_size == 0 or sequence_length == 0) return TrainerError.InvalidBatchSize;
        const token_count = std.math.mul(usize, batch_size, sequence_length) catch return TrainerError.ValueOverflow;
        const inputs = self.batch_buffers.inputs[0..token_count];
        const targets = self.batch_buffers.targets[0..token_count];
        const lengths = self.batch_buffers.lengths[0..batch_size];
        @memset(inputs, 0);
        @memset(targets, 0);
        @memset(lengths, sequence_length);
        var row: usize = 0;
        while (row < batch_size) : (row += 1) {
            const local_position = std.math.add(usize, batch_offset, row) catch return TrainerError.ValueOverflow;
            const position = std.math.add(usize, layout.rank_start, local_position) catch return TrainerError.ValueOverflow;
            const window: usize = @intCast(permuteWindowPosition(@intCast(position), @intCast(layout.selected_windows), permutation_key));
            const source_base = std.math.mul(usize, window, layout.window_size) catch return TrainerError.ValueOverflow;
            const source_end = std.math.add(usize, source_base, layout.window_size) catch return TrainerError.ValueOverflow;
            if (source_end > payload.len) return TrainerError.IndexOutOfBounds;
            const source = payload[source_base..source_end];
            const target_base = std.math.mul(usize, row, sequence_length) catch return TrainerError.ValueOverflow;
            const input_row = inputs[target_base .. target_base + sequence_length];
            const target_row = targets[target_base .. target_base + sequence_length];
            @memcpy(input_row, source[0..sequence_length]);
            @memcpy(target_row, source[1 .. sequence_length + 1]);
            for (input_row) |token| if (@as(usize, token) >= self.vocab_size) return TrainerError.TokenIndexOutOfRange;
            for (target_row) |token| if (@as(usize, token) >= self.vocab_size) return TrainerError.TokenIndexOutOfRange;
        }
        return PreparedBatch{
            .allocator = self.allocator,
            .token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator),
            .active_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator),
            .real_sequence_lengths = lengths,
            .flat_input_tokens = inputs,
            .flat_target_tokens = targets,
            .effective_batch_size = batch_size,
            .sequence_length = sequence_length,
            .local_active_samples = batch_size,
            .local_token_count = token_count,
            .pooled = true,
        };
    }
    pub fn epochBatchCount(self: *DistributedTrainerFuthark, dataset: *const mmap_dataset.MmapTokenDataset) !u64 {
        const layout = try self.tokenEpochLayout(dataset, 0, 0);
        return layout.global_batch_count;
    }
    pub fn trainEpochTokens(self: *DistributedTrainerFuthark, dataset: *const mmap_dataset.MmapTokenDataset, epoch: u64) !f32 {
        return self.trainEpochTokensFrom(dataset, epoch, 0);
    }
    pub fn trainEpochTokensFrom(
        self: *DistributedTrainerFuthark,
        dataset: *const mmap_dataset.MmapTokenDataset,
        epoch: u64,
        first_batch: u64,
    ) !f32 {
        const layout = try self.tokenEpochLayout(dataset, epoch, first_batch);
        const permutation_key = windowPermutationKey(self.config.embedding_seed, epoch);
        var weighted_loss: f64 = 0.0;
        var weight: f64 = 0.0;
        var index: u64 = first_batch;
        while (index < layout.global_batch_count) : (index += 1) {
            var prepare_failure: ?anyerror = null;
            var prepared_opt: ?PreparedBatch = null;
            if (index < layout.local_batch_count) {
                prepared_opt = captureFailure(&prepare_failure, self.prepareTokenBatch(dataset.payload, layout, index, permutation_key));
            } else {
                const empty_batch: [][]const u8 = &.{};
                prepared_opt = captureFailure(&prepare_failure, self.prepareBatch(empty_batch));
            }
            try self.settlePreparedBatch(prepare_failure, &prepared_opt);
            var prepared = prepared_opt orelse return TrainerError.InvalidTrainingState;
            defer prepared.deinit();
            var no_next: ?PreparedBatch = null;
            const result = try self.trainPreparedStepFuthark(&prepared, null, &no_next);
            if (no_next) |*next| next.deinit();
            if (result.sample_weight > 0.0) {
                weighted_loss += @as(f64, result.loss) * result.sample_weight;
                weight += result.sample_weight;
            }
        }
        if (weight == 0.0) return 0.0;
        const loss: f32 = @floatCast(weighted_loss / weight);
        if (!std.math.isFinite(loss)) return TrainerError.InvalidLoss;
        return loss;
    }
    fn rankWindowBounds(total: usize, world_size: usize, rank: usize) !struct { start: usize, count: usize } {
        if (world_size == 0) return TrainerError.InvalidWorldSize;
        if (rank >= world_size) return TrainerError.InvalidRank;
        const base = total / world_size;
        const remainder = total % world_size;
        const count = base + @intFromBool(rank < remainder);
        const start = if (rank < remainder)
            std.math.mul(usize, rank, base + 1) catch return TrainerError.ValueOverflow
        else
            std.math.add(
                usize,
                std.math.mul(usize, remainder, base + 1) catch return TrainerError.ValueOverflow,
                std.math.mul(usize, rank - remainder, base) catch return TrainerError.ValueOverflow,
            ) catch return TrainerError.ValueOverflow;
        return .{ .start = start, .count = count };
    }
    fn prepareBatch(self: *DistributedTrainerFuthark, batch: [][]const u8) !PreparedBatch {
        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        errdefer {
            for (token_lists.items) |*list| list.deinit();
            token_lists.deinit();
        }
        const sequence_length = self.runtime_max_seq_len;
        const max_token_count = try std.math.add(usize, sequence_length, 1);
        const token_bound_bytes = try std.math.mul(usize, max_token_count, 4);
        const maximum_input_bytes = @min(token_bound_bytes, self.runtime_tokenizer_max_input_bytes);
        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            const bounded_text = truncateUtf8Input(text, maximum_input_bytes);
            self.tokenizer.encodeBounded(bounded_text, max_token_count, &token_list) catch |err| {
                token_list.deinit();
                return err;
            };
            token_lists.append(token_list) catch |err| {
                token_list.deinit();
                return err;
            };
        }
        var local_active_samples: u64 = 0;
        var local_token_count: u64 = 0;
        for (token_lists.items) |*list| {
            if (list.items.len > max_token_count) list.shrinkRetainingCapacity(max_token_count);
            if (list.items.len >= 2) {
                local_active_samples = try std.math.add(u64, local_active_samples, 1);
                local_token_count = try std.math.add(u64, local_token_count, @intCast(list.items.len - 1));
            }
        }
        var active_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        errdefer active_lists.deinit();
        for (token_lists.items) |list| {
            if (list.items.len >= 2) try active_lists.append(list);
        }
        const effective_batch_size = @max(active_lists.items.len, @as(usize, 1));
        const real_sequence_lengths = try self.allocator.alloc(usize, effective_batch_size);
        errdefer self.allocator.free(real_sequence_lengths);
        @memset(real_sequence_lengths, 0);
        for (active_lists.items, 0..) |token_list, index| {
            real_sequence_lengths[index] = @min(token_list.items.len - 1, sequence_length);
        }
        const compact_seq = compact_batch.compactSequenceLength(real_sequence_lengths);
        const flat_size = try compact_batch.packedTokenCount(effective_batch_size, compact_seq);
        const flat_input_tokens = try self.allocator.alloc(u32, flat_size);
        errdefer self.allocator.free(flat_input_tokens);
        const flat_target_tokens = try self.allocator.alloc(u32, flat_size);
        errdefer self.allocator.free(flat_target_tokens);
        @memset(flat_input_tokens, 0);
        @memset(flat_target_tokens, 0);
        for (active_lists.items, 0..) |token_list, batch_index| {
            const prediction_length = real_sequence_lengths[batch_index];
            var sequence_index: usize = 0;
            while (sequence_index < prediction_length) : (sequence_index += 1) {
                const flat_index = try std.math.add(
                    usize,
                    try std.math.mul(usize, batch_index, compact_seq),
                    sequence_index,
                );
                const input_token = token_list.items[sequence_index];
                const target_token = token_list.items[sequence_index + 1];
                if (@as(usize, input_token) >= self.vocab_size or @as(usize, target_token) >= self.vocab_size) return TrainerError.TokenIndexOutOfRange;
                flat_input_tokens[flat_index] = input_token;
                flat_target_tokens[flat_index] = target_token;
            }
        }
        if (self.config.shuffle_target_control) {
            self.shuffle_mutex.lock();
            var local_shuffle_state = self.shuffle_control_state;
            self.shuffle_mutex.unlock();
            var permute_index: usize = flat_target_tokens.len;
            while (permute_index > 1) {
                permute_index -= 1;
                local_shuffle_state = local_shuffle_state *% 6364136223846793005 +% 1442695040888963407;
                const draw: usize = @intCast((local_shuffle_state >> 33) % @as(u64, @intCast(permute_index + 1)));
                const swap = flat_target_tokens[permute_index];
                flat_target_tokens[permute_index] = flat_target_tokens[draw];
                flat_target_tokens[draw] = swap;
            }
            self.shuffle_mutex.lock();
            self.shuffle_control_state = local_shuffle_state;
            self.shuffle_mutex.unlock();
        }
        return PreparedBatch{
            .allocator = self.allocator,
            .token_lists = token_lists,
            .active_lists = active_lists,
            .real_sequence_lengths = real_sequence_lengths,
            .flat_input_tokens = flat_input_tokens,
            .flat_target_tokens = flat_target_tokens,
            .effective_batch_size = effective_batch_size,
            .sequence_length = compact_seq,
            .local_active_samples = local_active_samples,
            .local_token_count = local_token_count,
        };
    }
    const BatchPreparationTask = struct {
        trainer: *DistributedTrainerFuthark,
        batch: [][]const u8,
        result: ?PreparedBatch = null,
        failure: ?anyerror = null,
        fn run(self: *BatchPreparationTask) void {
            self.result = self.trainer.prepareBatch(self.batch) catch |err| {
                self.failure = err;
                return;
            };
        }
    };
    fn accumulateEmbeddingGradientsFromDelta(
        self: *DistributedTrainerFuthark,
        flat_input_tokens: []const u32,
        real_sequence_lengths: []const usize,
        input_delta: *FutharkArray3DF16,
    ) !void {
        if (self.gpu_embedding == null or flat_input_tokens.len == 0) return;
        const embedding = &self.gpu_embedding.?;
        if (input_delta.dim2 != embedding.dim) return TrainerError.InvalidWeightsShape;
        const expected_rows = try std.math.mul(usize, input_delta.dim0, input_delta.dim1);
        if (expected_rows != flat_input_tokens.len or input_delta.dim0 != real_sequence_lengths.len) return TrainerError.InvalidWeightsShape;
        const context = &self.accelerator.ctx;
        context.mutex.lock();
        defer context.mutex.unlock();
        try embedding.backwardPaddedAccumulate(
            flat_input_tokens,
            real_sequence_lengths,
            input_delta,
        );
    }
    fn reduceStackGradients(self: *DistributedTrainerFuthark, fused: *accel.FusedStepResult, pre_failure: ?anyerror) !void {
        var failure = pre_failure;
        const distributed = self.coordinator.world_size > 1;
        const ctx = &self.accelerator.ctx;
        var views = if (failure == null and distributed) captureFailure(&failure, fused.gradientDeviceBuffers()) else null;
        defer if (views) |*held| {
            held.*[0].deinit();
            held.*[1].deinit();
        };
        if (views) |*held| {
            if (captureFailure(&failure, held.*[0].requireSameContext(ctx)) != null) {
                _ = captureFailure(&failure, held.*[1].requireSameContext(ctx));
            }
        }
        const pointer_s = if (failure == null) (if (views) |*held| captureFailure(&failure, held.*[0].rawPointer()) else null) else null;
        const pointer_t = if (failure == null) (if (views) |*held| captureFailure(&failure, held.*[1].rawPointer()) else null) else null;
        if (try self.agreeOnStatus(failure != null)) return failure orelse TrainerError.PeerRankFailure;
        if (!distributed) return;
        if (views) |*held| {
            {
                self.nccl_mutex.lock();
                defer self.nccl_mutex.unlock();
                try self.coordinator.allReduceFloat32(pointer_s.?, pointer_s.?, held.*[0].elementCount());
                try self.coordinator.allReduceFloat32(pointer_t.?, pointer_t.?, held.*[1].elementCount());
                try self.coordinator.synchronize();
            }
            held.*[0].deinit();
            held.*[1].deinit();
            views = null;
        }
    }
    fn updateEmbeddingFromJob(self: *DistributedTrainerFuthark, job: *const StepUpdateJob, failure: *?anyerror) !void {
        const ctx = &self.accelerator.ctx;
        const distributed = self.coordinator.world_size > 1;
        const active = job.apply_embedding_update and self.gpu_embedding != null;
        if (failure.* == null and active and distributed) {
            const embedding = &self.gpu_embedding.?;
            {
                ctx.mutex.lock();
                defer ctx.mutex.unlock();
                const gradient_scale: f32 = if (self.config.grad_mean) job.local_fraction else 1.0;
                _ = captureFailure(failure, embedding.scaleGradient(gradient_scale));
            }
            if (failure.* == null) _ = captureFailure(failure, ctx.sync());
        }
        var gradient = if (failure.* == null and active and distributed) blk: {
            ctx.mutex.lock();
            defer ctx.mutex.unlock();
            break :blk captureFailure(failure, self.gpu_embedding.?.getGradientDevicePtrF32());
        } else null;
        defer if (gradient) |*held| held.deinit();
        if (gradient) |*held| {
            _ = captureFailure(failure, held.requireSameContext(ctx));
        }
        const gradient_pointer = if (failure.* == null) (if (gradient) |*held| captureFailure(failure, held.rawPointer()) else null) else null;
        if (try self.agreeOnStatus(failure.* != null)) return failure.* orelse TrainerError.PeerRankFailure;
        if (gradient) |*held| {
            {
                self.nccl_mutex.lock();
                defer self.nccl_mutex.unlock();
                try self.coordinator.allReduceFloat32(gradient_pointer.?, gradient_pointer.?, held.elementCount());
                try self.coordinator.synchronize();
            }
            held.deinit();
            gradient = null;
        }
        if (active) {
            ctx.mutex.lock();
            defer ctx.mutex.unlock();
            const embedding = &self.gpu_embedding.?;
            if (self.config.use_normalized_gradient_flow) _ = captureFailure(failure, embedding.clipGradient(self.config.gradient_clip_norm));
            if (failure.* == null) {
                _ = captureFailure(failure, embedding.applyUpdateFusedSFD(
                    job.learning_rate,
                    job.momentum_beta,
                    job.fisher_gamma,
                    job.fisher_epsilon,
                    self.config.trust_ratio,
                    self.config.weight_floor,
                ));
            }
        }
    }
    fn trainPreparedStepFuthark(
        self: *DistributedTrainerFuthark,
        prepared: *PreparedBatch,
        next_batch: ?[][]const u8,
        next_prepared: *?PreparedBatch,
    ) !StepResult {
        next_prepared.* = null;
        try self.absorbStepState();
        const global_active_samples = try self.allReduceSumU64(prepared.local_active_samples);
        if (global_active_samples == 0) {
            try self.prepareNextBatchAgreed(next_batch, next_prepared);
            return StepResult{ .step = self.global_step, .loss = 0.0, .reconstruction_loss = 0.0, .source_rms = 0.0, .sample_weight = 0.0 };
        }
        const global_token_count = if (self.coordinator.world_size > 1)
            try self.allReduceSumU64(prepared.local_token_count)
        else
            prepared.local_token_count;
        if (global_token_count == 0) {
            try self.prepareNextBatchAgreed(next_batch, next_prepared);
            return StepResult{ .step = self.global_step, .loss = 0.0, .reconstruction_loss = 0.0, .source_rms = 0.0, .sample_weight = 0.0 };
        }
        const local_fraction: f32 = if (self.coordinator.world_size > 1)
            @floatCast(
                @as(f64, @floatFromInt(prepared.local_token_count)) /
                    @as(f64, @floatFromInt(global_token_count)),
            )
        else
            1.0;
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;
        const completed_step = std.math.add(u64, self.global_step, 1) catch return TrainerError.ValueOverflow;
        const warmup_factor: f32 = if (self.config.optimizer_warmup_steps > 0 and completed_step < self.config.optimizer_warmup_steps)
            @as(f32, @floatFromInt(completed_step)) / @as(f32, @floatFromInt(self.config.optimizer_warmup_steps))
        else
            1.0;
        const learning_rate = self.learning_rate * warmup_factor;
        if (!std.math.isFinite(learning_rate) or learning_rate <= 0.0) return TrainerError.InvalidLearningRate;
        const report_progress = self.coordinator.isRoot() and (completed_step <= 50 or completed_step % 10 == 0);
        const step_t0_ns = std.time.nanoTimestamp();
        if (report_progress) {
            std.debug.print(
                "[Rank 0] Step {d} start batch={d} seq={d} dim={d} layers={d} tokens={d}\n",
                .{ completed_step, prepared.effective_batch_size, prepared.sequence_length, self.model_dim, self.num_layers, prepared.local_token_count },
            );
        }
        const step_for_phase = self.global_step;
        const effective_reconstruction_alpha: f32 = blk: {
            if (self.config.phase_a_steps > 0 and step_for_phase < self.config.phase_a_steps) break :blk 1.0;
            const ramp_span = self.config.phase_b_steps;
            if (ramp_span == 0) break :blk self.config.reconstruction_alpha;
            const ramp_end = std.math.add(u64, self.config.phase_a_steps, ramp_span) catch break :blk self.config.reconstruction_alpha;
            if (step_for_phase >= ramp_end) break :blk self.config.reconstruction_alpha;
            const elapsed = step_for_phase - self.config.phase_a_steps;
            const progress = @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(ramp_span));
            const value = 1.0 - progress * (1.0 - self.config.reconstruction_alpha);
            if (!std.math.isFinite(value)) break :blk self.config.reconstruction_alpha;
            break :blk value;
        };
        const clamped_reconstruction_alpha = @max(@as(f32, 0.0), @min(@as(f32, 1.0), effective_reconstruction_alpha));
        const BatchTensors = struct {
            inputs: FutharkArray3DF16,
            targets: FutharkArray3DF16,
        };
        var local_failure: ?anyerror = null;
        var tensors: ?BatchTensors = null;
        defer if (tensors) |*held| {
            held.inputs.free(&self.accelerator.ctx) catch |err| std.log.err("distributed trainer: releasing padded input embeddings failed: {s}", .{@errorName(err)});
            held.targets.free(&self.accelerator.ctx) catch |err| std.log.err("distributed trainer: releasing padded target embeddings failed: {s}", .{@errorName(err)});
        };
        if (self.gpu_embedding) |*embedding| {
            const context = &self.accelerator.ctx;
            context.mutex.lock();
            defer context.mutex.unlock();
            const forward_inputs = captureFailure(&local_failure, embedding.forwardPadded(prepared.flat_input_tokens, prepared.real_sequence_lengths, prepared.sequence_length));
            if (forward_inputs) |input_value| {
                var input_handle = input_value;
                const forward_targets = if (self.target_source) |*frozen_source|
                    captureFailure(&local_failure, frozen_source.forwardPadded(prepared.flat_target_tokens, prepared.real_sequence_lengths, prepared.sequence_length))
                else
                    captureFailure(&local_failure, embedding.forwardPadded(prepared.flat_target_tokens, prepared.real_sequence_lengths, prepared.sequence_length));
                if (forward_targets) |target_value| {
                    tensors = BatchTensors{ .inputs = input_handle, .targets = target_value };
                } else {
                    input_handle.free(context) catch |err| std.log.err("distributed trainer: releasing padded input embeddings failed: {s}", .{@errorName(err)});
                }
            }
        } else {
            local_failure = TrainerError.InvalidTrainingState;
        }
        var preparation_task: ?BatchPreparationTask = if (next_batch) |batch|
            BatchPreparationTask{ .trainer = self, .batch = batch }
        else
            null;
        var preparation_thread: ?std.Thread = null;
        defer {
            if (preparation_thread) |thread| thread.join();
            if (preparation_task) |*task| {
                if (task.result) |*value| value.deinit();
            }
        }
        if (local_failure == null) {
            if (preparation_task) |*task| {
                preparation_thread = std.Thread.spawn(.{}, BatchPreparationTask.run, .{task}) catch |err| blk: {
                    local_failure = err;
                    break :blk null;
                };
            }
        }
        var fused_result: ?accel.FusedStepResult = null;
        defer if (fused_result) |*value| value.deinit();
        if (local_failure == null) {
            if (tensors) |*held| {
                fused_result = captureFailure(&local_failure, self.accelerator.fusedTrainingStep(
                    &held.inputs,
                    &held.targets,
                    prepared.real_sequence_lengths,
                    self.config.grad_mean,
                    if (self.config.grad_mean) local_fraction else 1.0,
                    clamped_reconstruction_alpha,
                    @as(f32, 1.0),
                    self.config.logdet_weight,
                ));
            }
        }
        if (local_failure == null) {
            if (fused_result) |*value| {
                _ = captureFailure(&local_failure, self.accumulateEmbeddingGradientsFromDelta(
                    prepared.flat_input_tokens,
                    prepared.real_sequence_lengths,
                    &value.input_delta,
                ));
            }
        }
        if (try self.agreeOnStatus(local_failure != null)) return local_failure orelse TrainerError.PeerRankFailure;
        const step_backward_ns = std.time.nanoTimestamp() - step_t0_ns;
        if (report_progress) std.debug.print("[Rank 0] Step {d} RSF/OFTB reversible backward gradients computed dt={d}ms\n", .{ completed_step, @divTrunc(step_backward_ns, 1_000_000) });
        const apply_spectral = self.gpu_embedding != null and self.config.spectral_iterations > 0 and completed_step % self.config.spectral_interval == 0;
        if (report_progress) std.debug.print("[Rank 0] Step {d} gradients accumulated, reducing current-step gradients\n", .{completed_step});
        const step_increment_value: u64 = 1;
        const job_fused = fused_result orelse return TrainerError.InvalidTrainingState;
        fused_result = null;
        const update_started = std.time.nanoTimestamp();
        try self.step_synchronizer.processStep(self, StepUpdateJob{
            .step = completed_step,
            .local_fraction = local_fraction,
            .learning_rate = learning_rate,
            .momentum_beta = self.momentum,
            .fisher_gamma = self.config.fisher_gamma,
            .fisher_epsilon = self.config.fisher_epsilon,
            .apply_embedding_update = self.gpu_embedding != null,
            .apply_spectral = apply_spectral,
            .local_step_increment = step_increment_value,
            .fused = job_fused,
        });
        const update_elapsed = std.time.nanoTimestamp() - update_started;
        if (report_progress) std.debug.print("[Rank 0] Step {d} reduction_update_ms={d}\n", .{ completed_step, @divTrunc(update_elapsed, std.time.ns_per_ms) });
        try self.absorbStepState();
        const telemetry = self.last_step_telemetry;
        if (!telemetry.finalized or telemetry.step != completed_step) return TrainerError.InvalidTrainingState;
        if (!std.math.isFinite(telemetry.loss) or !std.math.isFinite(telemetry.reconstruction_loss)) return TrainerError.InvalidLoss;
        if (report_progress) {
            const step_total_elapsed = std.time.nanoTimestamp() - step_t0_ns;
            std.debug.print(
                "[Rank 0] Step {d} completed loss={d:.6} recon={d:.6} logdet={d:.6} alpha={d:.4} src_rms={d:.6} global_tokens={d} step_total_ms={d}\n",
                .{ completed_step, telemetry.loss, telemetry.reconstruction_loss, telemetry.logdet_mean, clamped_reconstruction_alpha, telemetry.source_rms, global_token_count, @divTrunc(step_total_elapsed, std.time.ns_per_ms) },
            );
        }
        var next_prepare_error: ?anyerror = null;
        if (preparation_thread) |thread| {
            thread.join();
            preparation_thread = null;
        }
        if (preparation_task) |*task| {
            next_prepare_error = task.failure;
            next_prepared.* = task.result;
            task.result = null;
        }
        try self.settlePreparedBatch(next_prepare_error, next_prepared);
        return StepResult{
            .step = telemetry.step,
            .loss = telemetry.loss,
            .reconstruction_loss = telemetry.reconstruction_loss,
            .source_rms = telemetry.source_rms,
            .sample_weight = @as(f64, @floatFromInt(prepared.local_token_count)),
            .logdet_mean = telemetry.logdet_mean,
        };
    }
    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !StepResult {
        var prepared = try self.prepareBatchAgreed(batch);
        defer prepared.deinit();
        var unused_next: ?PreparedBatch = null;
        defer if (unused_next) |*value| value.deinit();
        return self.trainPreparedStepFuthark(&prepared, null, &unused_next);
    }
    fn makeTmpFilePath(
        self: *DistributedTrainerFuthark,
        suffix: []const u8,
    ) ![]u8 {
        const directory = try resolveTemporaryDirectory(self.allocator);
        defer self.allocator.free(directory);
        const timestamp = std.time.nanoTimestamp();
        return std.fmt.allocPrint(self.allocator, "{s}{c}jaide_{s}_{d}_{d}.tmp", .{ directory, std.fs.path.sep, suffix, self.coordinator.rank, timestamp });
    }
    fn readWholeFile(self: *DistributedTrainerFuthark, path: []const u8, max_size: usize) ![]u8 {
        const file = try openReadFile(path);
        defer file.close();
        const length_u64 = try file.getEndPos();
        const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
        if (length == 0 or length > max_size) return TrainerError.FileTooLarge;
        const data = try self.allocator.alloc(u8, length);
        errdefer self.allocator.free(data);
        try file.reader().readNoEof(data);
        return data;
    }
    fn writeNsirGraph(writer: anytype, graph: *SelfSimilarRelationalGraph, config: TrainerConfig) !void {
        if (graph.nodes.count() > config.max_node_count) return TrainerError.NodeCountTooLarge;
        const node_count = try safeUsizeToU32(graph.nodes.count());
        try writer.writeInt(u32, node_count, .little);
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr.*;
            if (node.id.len > config.max_id_length) return TrainerError.NodeIdTooLong;
            if (node.data.len > config.max_node_data_length) return TrainerError.NodeDataTooLong;
            if (!std.math.isFinite(node.qubit.a.re) or !std.math.isFinite(node.qubit.a.im) or
                !std.math.isFinite(node.qubit.b.re) or !std.math.isFinite(node.qubit.b.im) or
                !std.math.isFinite(node.phase)) return TrainerError.InvalidQuantumState;
            const id_len = try safeUsizeToU32(node.id.len);
            try writer.writeInt(u32, id_len, .little);
            try writer.writeAll(node.id);
            const data_len = try safeUsizeToU32(node.data.len);
            try writer.writeInt(u32, data_len, .little);
            try writer.writeAll(node.data);
            try writeF64(writer, node.qubit.a.re);
            try writeF64(writer, node.qubit.a.im);
            try writeF64(writer, node.qubit.b.re);
            try writeF64(writer, node.qubit.b.im);
            try writeF64(writer, node.phase);
        }
        const edge_key_count = try safeUsizeToU32(graph.edges.count());
        if (edge_key_count > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
        try writer.writeInt(u32, edge_key_count, .little);
        var edge_iter = graph.edges.iterator();
        var total_edges: u64 = 0;
        while (edge_iter.next()) |entry| {
            const key = entry.key_ptr.*;
            const edge_list = entry.value_ptr.*;
            if (key.source.len > config.max_id_length) return TrainerError.NodeIdTooLong;
            if (key.target.len > config.max_id_length) return TrainerError.NodeIdTooLong;
            if (edge_list.items.len > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
            total_edges = try std.math.add(u64, total_edges, @as(u64, @intCast(edge_list.items.len)));
            if (total_edges > @as(u64, config.max_edge_group_count)) return TrainerError.EdgeCountTooLarge;
            const src_len = try safeUsizeToU32(key.source.len);
            try writer.writeInt(u32, src_len, .little);
            try writer.writeAll(key.source);
            const tgt_len = try safeUsizeToU32(key.target.len);
            try writer.writeInt(u32, tgt_len, .little);
            try writer.writeAll(key.target);
            const count = try safeUsizeToU32(edge_list.items.len);
            try writer.writeInt(u32, count, .little);
            for (edge_list.items) |edge| {
                if (!std.math.isFinite(edge.weight)) return TrainerError.InvalidEdgeWeight;
                if (!std.math.isFinite(edge.quantum_correlation.re) or !std.math.isFinite(edge.quantum_correlation.im)) return TrainerError.InvalidEdgeWeight;
                if (!std.math.isFinite(edge.fractal_dimension)) return TrainerError.InvalidEdgeWeight;
                try writeF64(writer, edge.weight);
                try writer.writeByte(@intFromEnum(edge.quality));
                try writeF64(writer, edge.quantum_correlation.re);
                try writeF64(writer, edge.quantum_correlation.im);
                try writeF64(writer, edge.fractal_dimension);
            }
        }
    }
    fn readNsirGraph(
        allocator: std.mem.Allocator,
        reader: anytype,
        config: TrainerConfig,
    ) !*SelfSimilarRelationalGraph {
        const graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var graph_ptr_committed = false;
        errdefer if (!graph_ptr_committed) allocator.destroy(graph_ptr);
        graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var graph_committed = false;
        errdefer if (!graph_committed) graph_ptr.deinit();
        const node_count = try reader.readInt(u32, .little);
        if (@as(u64, node_count) > @as(u64, config.max_node_count)) return TrainerError.NodeCountTooLarge;
        var ni: u32 = 0;
        while (ni < node_count) : (ni += 1) {
            const id_len = try reader.readInt(u32, .little);
            if (id_len > config.max_id_length) return TrainerError.NodeIdTooLong;
            const id = try allocator.alloc(u8, id_len);
            defer allocator.free(id);
            try reader.readNoEof(id);
            const data_len = try reader.readInt(u32, .little);
            if (data_len > config.max_node_data_length) return TrainerError.NodeDataTooLong;
            const data_bytes = try allocator.alloc(u8, data_len);
            defer allocator.free(data_bytes);
            try reader.readNoEof(data_bytes);
            const a_re = try readF64(reader);
            const a_im = try readF64(reader);
            const b_re = try readF64(reader);
            const b_im = try readF64(reader);
            const phase = try readF64(reader);
            if (!std.math.isFinite(a_re) or !std.math.isFinite(a_im) or !std.math.isFinite(b_re) or !std.math.isFinite(b_im) or !std.math.isFinite(phase)) return TrainerError.InvalidQuantumState;
            const qubit = nsir.Qubit.init(
                std.math.Complex(f64).init(a_re, a_im),
                std.math.Complex(f64).init(b_re, b_im),
            );
            const node = try nsir.Node.init(graph_ptr.allocator, id, data_bytes, qubit, phase);
            try graph_ptr.addNode(node);
        }
        const edge_key_count = try reader.readInt(u32, .little);
        if (edge_key_count > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
        var ei: u32 = 0;
        var total_edges: u64 = 0;
        while (ei < edge_key_count) : (ei += 1) {
            const src_len = try reader.readInt(u32, .little);
            if (src_len > config.max_id_length) return TrainerError.NodeIdTooLong;
            const source = try allocator.alloc(u8, src_len);
            defer allocator.free(source);
            try reader.readNoEof(source);
            const tgt_len = try reader.readInt(u32, .little);
            if (tgt_len > config.max_id_length) return TrainerError.NodeIdTooLong;
            const target = try allocator.alloc(u8, tgt_len);
            defer allocator.free(target);
            try reader.readNoEof(target);
            const count = try reader.readInt(u32, .little);
            if (count > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
            total_edges = try std.math.add(u64, total_edges, @as(u64, count));
            if (total_edges > @as(u64, config.max_edge_group_count)) return TrainerError.EdgeCountTooLarge;
            var k: u32 = 0;
            while (k < count) : (k += 1) {
                const weight = try readF64(reader);
                const quality_byte = try reader.readByte();
                const quality: nsir.EdgeQuality = std.meta.intToEnum(nsir.EdgeQuality, quality_byte) catch return TrainerError.InvalidQualityByte;
                const qc_re = try readF64(reader);
                const qc_im = try readF64(reader);
                const fd = try readF64(reader);
                if (!std.math.isFinite(weight) or !std.math.isFinite(qc_re) or !std.math.isFinite(qc_im) or !std.math.isFinite(fd)) return TrainerError.InvalidEdgeWeight;
                const edge = try nsir.Edge.init(
                    graph_ptr.allocator,
                    source,
                    target,
                    quality,
                    weight,
                    std.math.Complex(f64).init(qc_re, qc_im),
                    fd,
                );
                try graph_ptr.addEdge(source, target, edge);
            }
        }
        graph_ptr_committed = true;
        graph_committed = true;
        return graph_ptr;
    }
    pub const CheckpointSnapshot = struct {
        allocator: std.mem.Allocator,
        checkpoint_version: u32,
        global_step: u64,
        model_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        local_batch_size: usize,
        learning_rate: f32,
        momentum: f32,
        clip_min_f32: f32,
        clip_max_f32: f32,
        fisher_gamma: f32,
        fisher_epsilon: f32,
        trust_ratio: f32,
        weight_floor: f32,
        optimizer_warmup_steps: u64,
        spectral_interval: u64,
        spectral_target_norm: f32,
        spectral_iterations: usize,
        reconstruction_alpha: f32,
        phase_a_steps: u64,
        phase_b_steps: u64,
        logdet_weight: f32,
        gradient_clip_norm: f32,
        grad_mean: bool,
        use_normalized_gradient_flow: bool,
        embedding_seed: u64,
        default_max_seq_len: usize,
        reasoning_cycles: usize,
        relational_pass_interval: usize,
        shuffle_target_control: bool,
        target_source_frozen: bool,
        spectral_depth_compensation: bool,
        shuffle_control_state: u64,
        relational_fast_mode: bool,
        rsf_optimizer_state: accel.RSFOptimizerState,
        embedding_optimizer_state: ?accel.EmbeddingOptimizerState,
        embedding_vocab: usize,
        embedding_dim: usize,
        target_vocab: usize,
        target_dim: usize,
        target_master_weights: []f32,
        knowledge_graph_nonce: [32]u8,
        training_graph_bytes: []u8,
        knowledge_graph_bytes: []u8,
        tokenizer_data: []u8,
        pub fn deinit(self: *CheckpointSnapshot) void {
            self.rsf_optimizer_state.deinit();
            if (self.embedding_optimizer_state) |*state| state.deinit();
            if (self.target_master_weights.len > 0) self.allocator.free(self.target_master_weights);
            self.allocator.free(self.training_graph_bytes);
            self.allocator.free(self.knowledge_graph_bytes);
            if (self.tokenizer_data.len > 0) self.allocator.free(self.tokenizer_data);
        }
    };
    pub fn captureCheckpointSnapshot(self: *DistributedTrainerFuthark) !*CheckpointSnapshot {
        if (!self.coordinator.isRoot()) return TrainerError.CheckpointSaveMustRunOnRoot;
        try self.absorbStepState();
        try self.accelerator.sync();
        const tokenizer_tmp = try self.makeTmpFilePath("tokenizer");
        defer self.allocator.free(tokenizer_tmp);
        var tokenizer_tmp_committed = false;
        defer if (!tokenizer_tmp_committed) deletePath(tokenizer_tmp);
        try self.tokenizer.saveVocab(tokenizer_tmp);
        const tokenizer_data = try self.readWholeFile(tokenizer_tmp, self.config.max_tokenizer_file_size);
        errdefer self.allocator.free(tokenizer_data);
        tokenizer_tmp_committed = true;
        deletePath(tokenizer_tmp);
        var rsf_optimizer_state = try self.accelerator.readOptimizerState(self.allocator);
        errdefer rsf_optimizer_state.deinit();
        var embedding_optimizer_state: ?accel.EmbeddingOptimizerState = null;
        errdefer if (embedding_optimizer_state) |*state| state.deinit();
        const clip_min_f32: f32 = @floatCast(self.accelerator.clip_min);
        const clip_max_f32: f32 = @floatCast(self.accelerator.clip_max);
        if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or !(clip_min_f32 < clip_max_f32)) return TrainerError.InvalidClipRange;
        var embedding_vocab: usize = 0;
        var embedding_dim: usize = 0;
        if (self.gpu_embedding) |*embedding| {
            embedding_vocab = embedding.vocab_size;
            embedding_dim = embedding.dim;
            const context = &self.accelerator.ctx;
            context.mutex.lock();
            defer context.mutex.unlock();
            embedding_optimizer_state = try embedding.readOptimizerState(self.allocator);
        }
        var target_vocab: usize = 0;
        var target_dim: usize = 0;
        var target_master_weights: []f32 = &.{};
        errdefer if (target_master_weights.len > 0) self.allocator.free(target_master_weights);
        if (self.target_source) |*target| {
            target_vocab = target.vocab_size;
            target_dim = target.dim;
            const context = &self.accelerator.ctx;
            context.mutex.lock();
            defer context.mutex.unlock();
            target_master_weights = try target.exportAsF32(self.allocator);
        }
        var training_graph_buffer = std.ArrayList(u8).init(self.allocator);
        errdefer training_graph_buffer.deinit();
        try writeNsirGraph(training_graph_buffer.writer(), self.nsir_graph, self.config);
        const training_graph_bytes = try training_graph_buffer.toOwnedSlice();
        errdefer self.allocator.free(training_graph_bytes);
        var knowledge_graph_buffer = std.ArrayList(u8).init(self.allocator);
        errdefer knowledge_graph_buffer.deinit();
        try writeNsirGraph(knowledge_graph_buffer.writer(), self.knowledge_nsir_graph, self.config);
        const knowledge_graph_bytes = try knowledge_graph_buffer.toOwnedSlice();
        errdefer self.allocator.free(knowledge_graph_bytes);
        self.shuffle_mutex.lock();
        const shuffle_control_state = self.shuffle_control_state;
        self.shuffle_mutex.unlock();
        const snapshot = try self.allocator.create(CheckpointSnapshot);
        errdefer self.allocator.destroy(snapshot);
        snapshot.* = CheckpointSnapshot{
            .allocator = self.allocator,
            .checkpoint_version = self.config.checkpoint_version,
            .global_step = self.global_step,
            .model_dim = self.model_dim,
            .num_layers = self.num_layers,
            .vocab_size = self.vocab_size,
            .local_batch_size = self.local_batch_size,
            .learning_rate = self.learning_rate,
            .momentum = self.momentum,
            .clip_min_f32 = clip_min_f32,
            .clip_max_f32 = clip_max_f32,
            .fisher_gamma = self.config.fisher_gamma,
            .fisher_epsilon = self.config.fisher_epsilon,
            .trust_ratio = self.config.trust_ratio,
            .weight_floor = self.config.weight_floor,
            .optimizer_warmup_steps = self.config.optimizer_warmup_steps,
            .spectral_interval = self.config.spectral_interval,
            .spectral_target_norm = self.config.spectral_target_norm,
            .spectral_iterations = self.config.spectral_iterations,
            .reconstruction_alpha = self.config.reconstruction_alpha,
            .phase_a_steps = self.config.phase_a_steps,
            .phase_b_steps = self.config.phase_b_steps,
            .logdet_weight = self.config.logdet_weight,
            .gradient_clip_norm = self.config.gradient_clip_norm,
            .grad_mean = self.config.grad_mean,
            .use_normalized_gradient_flow = self.config.use_normalized_gradient_flow,
            .embedding_seed = self.config.embedding_seed,
            .default_max_seq_len = self.config.default_max_seq_len,
            .reasoning_cycles = self.config.reasoning_cycles,
            .relational_pass_interval = self.config.relational_pass_interval,
            .shuffle_target_control = self.config.shuffle_target_control,
            .target_source_frozen = self.config.target_source_frozen,
            .spectral_depth_compensation = self.config.spectral_depth_compensation,
            .shuffle_control_state = shuffle_control_state,
            .relational_fast_mode = self.relational_fast_mode,
            .rsf_optimizer_state = rsf_optimizer_state,
            .embedding_optimizer_state = embedding_optimizer_state,
            .embedding_vocab = embedding_vocab,
            .embedding_dim = embedding_dim,
            .target_vocab = target_vocab,
            .target_dim = target_dim,
            .target_master_weights = target_master_weights,
            .knowledge_graph_nonce = self.knowledge_graph_nonce,
            .training_graph_bytes = training_graph_bytes,
            .knowledge_graph_bytes = knowledge_graph_bytes,
            .tokenizer_data = tokenizer_data,
        };
        return snapshot;
    }
    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const snapshot = try self.captureCheckpointSnapshot();
        defer {
            snapshot.deinit();
            self.allocator.destroy(snapshot);
        }
        try writeCheckpointSnapshotFile(snapshot, path);
        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, snapshot.global_step });
    }
    pub fn writeCheckpointSnapshotFile(snapshot: *const CheckpointSnapshot, path: []const u8) !void {
        const timestamp = std.time.nanoTimestamp();
        const checkpoint_tmp = try std.fmt.allocPrint(snapshot.allocator, "{s}.checkpoint.{d}.tmp", .{ path, timestamp });
        defer snapshot.allocator.free(checkpoint_tmp);
        var checkpoint_committed = false;
        defer if (!checkpoint_committed) deletePath(checkpoint_tmp);
        {
            const file = try createWriteFile(checkpoint_tmp);
            var file_closed = false;
            defer if (!file_closed) file.close();
            var bw = std.io.bufferedWriter(file.writer());
            const BW = @TypeOf(bw);
            var crc_writer = CrcTrackingWriter(BW.Writer){
                .inner = bw.writer(),
                .crc = std.hash.Crc32.init(),
            };
            const writer = crc_writer.writer();
            try writer.writeAll(CHECKPOINT_MAGIC[0..]);
            try writer.writeInt(u32, snapshot.checkpoint_version, .little);
            try writer.writeInt(u32, checkpoint_envelope.ENDIAN_MARKER, .little);
            const metadata = checkpoint_envelope.Metadata{
                .global_step = snapshot.global_step,
                .model_dim = @intCast(snapshot.model_dim),
                .layer_count = @intCast(snapshot.num_layers),
                .vocab_size = @intCast(snapshot.vocab_size),
                .local_batch_size = @intCast(snapshot.local_batch_size),
                .learning_rate = snapshot.learning_rate,
                .momentum = snapshot.momentum,
                .fisher_gamma = snapshot.fisher_gamma,
                .fisher_epsilon = snapshot.fisher_epsilon,
                .trust_ratio = snapshot.trust_ratio,
                .weight_floor = snapshot.weight_floor,
                .optimizer_warmup_steps = snapshot.optimizer_warmup_steps,
                .spectral_interval = snapshot.spectral_interval,
                .spectral_target_norm = snapshot.spectral_target_norm,
                .spectral_iterations = @intCast(snapshot.spectral_iterations),
                .reconstruction_alpha = snapshot.reconstruction_alpha,
                .phase_a_steps = snapshot.phase_a_steps,
                .phase_b_steps = snapshot.phase_b_steps,
                .logdet_weight = snapshot.logdet_weight,
                .gradient_clip_norm = snapshot.gradient_clip_norm,
                .grad_mean = snapshot.grad_mean,
                .use_normalized_gradient_flow = snapshot.use_normalized_gradient_flow,
                .embedding_seed = snapshot.embedding_seed,
                .default_max_seq_len = @intCast(snapshot.default_max_seq_len),
                .reasoning_cycles = @intCast(snapshot.reasoning_cycles),
                .relational_pass_interval = @intCast(snapshot.relational_pass_interval),
                .shuffle_target_control = snapshot.shuffle_target_control,
                .target_source_frozen = snapshot.target_source_frozen,
                .spectral_depth_compensation = snapshot.spectral_depth_compensation,
                .shuffle_control_state = snapshot.shuffle_control_state,
                .relational_fast_mode = snapshot.relational_fast_mode,
                .clip_min = snapshot.clip_min_f32,
                .clip_max = snapshot.clip_max_f32,
                .rsf_optimizer_step = snapshot.rsf_optimizer_state.step,
            };
            try checkpoint_envelope.writeMetadata(writer, metadata);
            try writeF32Array(writer, snapshot.rsf_optimizer_state.master_weights_s, false);
            try writeF32Array(writer, snapshot.rsf_optimizer_state.master_weights_t, false);
            try writeF32Array(writer, snapshot.rsf_optimizer_state.momentum_s, false);
            try writeF32Array(writer, snapshot.rsf_optimizer_state.momentum_t, false);
            try writeF32Array(writer, snapshot.rsf_optimizer_state.fisher_s, true);
            try writeF32Array(writer, snapshot.rsf_optimizer_state.fisher_t, true);
            if (snapshot.embedding_optimizer_state) |embedding_state| {
                try writer.writeByte(1);
                try writer.writeInt(u64, @intCast(snapshot.embedding_vocab), .little);
                try writer.writeInt(u64, @intCast(snapshot.embedding_dim), .little);
                try writer.writeInt(u64, embedding_state.step, .little);
                try writeF32Array(writer, embedding_state.master_weights, false);
                try writeF32Array(writer, embedding_state.momentum, false);
                try writeF32Array(writer, embedding_state.fisher, true);
            } else {
                try writer.writeByte(0);
            }
            if (snapshot.target_master_weights.len > 0) {
                try writer.writeByte(1);
                try writer.writeInt(u64, @intCast(snapshot.target_vocab), .little);
                try writer.writeInt(u64, @intCast(snapshot.target_dim), .little);
                try writeF32Array(writer, snapshot.target_master_weights, false);
            } else {
                try writer.writeByte(0);
            }
            try writer.writeAll(snapshot.knowledge_graph_nonce[0..]);
            try writer.writeAll(snapshot.training_graph_bytes);
            try writer.writeAll(snapshot.knowledge_graph_bytes);
            try writer.writeInt(u64, @as(u64, snapshot.tokenizer_data.len), .little);
            try writer.writeAll(snapshot.tokenizer_data);
            try writer.writeInt(u32, CHECKPOINT_TRAILER, .little);
            const final_crc = crc_writer.crc.final();
            try bw.flush();
            try bw.writer().writeInt(u32, final_crc, .little);
            try bw.flush();
            try file.sync();
            file.close();
            file_closed = true;
        }
        try renamePath(checkpoint_tmp, path);
        checkpoint_committed = true;
        syncContainingDirectory(path);
    }
    fn readCheckpointF32Array(self: *DistributedTrainerFuthark, reader: anytype, expected_length: usize, nonnegative: bool) ![]f32 {
        const saved_length_u64 = try reader.readInt(u64, .little);
        const saved_length = std.math.cast(usize, saved_length_u64) orelse return TrainerError.InvalidWeightsShape;
        if (saved_length != expected_length) return TrainerError.InvalidWeightsShape;
        const values = try self.allocator.alloc(f32, saved_length);
        errdefer self.allocator.free(values);
        var bytes: [16 * 1024]u8 = undefined;
        var offset: usize = 0;
        while (offset < values.len) {
            const count = @min(bytes.len / @sizeOf(f32), values.len - offset);
            try reader.readNoEof(bytes[0 .. count * 4]);
            var index: usize = 0;
            while (index < count) : (index += 1) {
                const value: f32 = @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
                if (!std.math.isFinite(value) or (nonnegative and value < 0.0)) return TrainerError.InvalidWeightValue;
                values[offset + index] = value;
            }
            offset += count;
        }
        return values;
    }
    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        var failure: ?anyerror = null;
        self.loadCheckpointLocal(path) catch |err| {
            failure = err;
        };
        if (try self.agreeOnStatus(failure != null)) return failure orelse TrainerError.PeerRankFailure;
    }
    fn loadCheckpointLocal(self: *DistributedTrainerFuthark, path: []const u8) !void {
        try self.absorbStepState();
        const raw_data = blk: {
            const file = try openReadFile(path);
            defer file.close();
            const length_u64 = try file.getEndPos();
            const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
            if (length < 16) return TrainerError.CheckpointCorrupted;
            const data = try self.allocator.alloc(u8, length);
            errdefer self.allocator.free(data);
            try file.reader().readNoEof(data);
            break :blk data;
        };
        defer self.allocator.free(raw_data);
        const checkpoint_payload = checkpoint_envelope.validate(raw_data, CHECKPOINT_MAGIC, CHECKPOINT_VERSION) catch |err| switch (err) {
            error.Truncated, error.ChecksumMismatch, error.EndiannessMismatch => return TrainerError.CheckpointCorrupted,
            error.MagicMismatch => return TrainerError.CheckpointMagicMismatch,
            error.VersionMismatch => return TrainerError.CheckpointVersionMismatch,
        };
        var fbs = std.io.fixedBufferStream(checkpoint_payload);
        const reader = fbs.reader();
        var magic_buf: [8]u8 = undefined;
        try reader.readNoEof(magic_buf[0..]);
        if (!std.mem.eql(u8, magic_buf[0..], CHECKPOINT_MAGIC[0..])) return TrainerError.CheckpointMagicMismatch;
        const version = try reader.readInt(u32, .little);
        if (version != CHECKPOINT_VERSION) return TrainerError.CheckpointVersionMismatch;
        if (try reader.readInt(u32, .little) != checkpoint_envelope.ENDIAN_MARKER) return TrainerError.CheckpointCorrupted;
        const metadata = checkpoint_envelope.readMetadata(reader) catch |err| switch (err) {
            error.InvalidFloat, error.InvalidBoolean => return TrainerError.CheckpointCorrupted,
            else => return err,
        };
        const saved_global_step = metadata.global_step;
        const saved_model_dim = std.math.cast(usize, metadata.model_dim) orelse return TrainerError.ModelDimMismatch;
        const saved_num_layers = std.math.cast(usize, metadata.layer_count) orelse return TrainerError.NumLayersMismatch;
        const saved_vocab_size = std.math.cast(usize, metadata.vocab_size) orelse return TrainerError.VocabSizeMismatch;
        const saved_local_batch_size = std.math.cast(usize, metadata.local_batch_size) orelse return TrainerError.InvalidBatchSize;
        const saved_learning_rate = metadata.learning_rate;
        const saved_momentum = metadata.momentum;
        const saved_fisher_gamma = metadata.fisher_gamma;
        const saved_fisher_epsilon = metadata.fisher_epsilon;
        const saved_trust_ratio = metadata.trust_ratio;
        const saved_weight_floor = metadata.weight_floor;
        const saved_warmup_steps = metadata.optimizer_warmup_steps;
        const saved_spectral_interval = metadata.spectral_interval;
        const saved_spectral_target = metadata.spectral_target_norm;
        const saved_spectral_iterations = std.math.cast(usize, metadata.spectral_iterations) orelse return TrainerError.InvalidSpectralState;
        const saved_reconstruction_alpha = metadata.reconstruction_alpha;
        const saved_phase_a_steps = metadata.phase_a_steps;
        const saved_phase_b_steps = metadata.phase_b_steps;
        const saved_logdet_weight = metadata.logdet_weight;
        const saved_gradient_clip_norm = metadata.gradient_clip_norm;
        const saved_grad_mean: u8 = @intFromBool(metadata.grad_mean);
        const saved_use_normalized_gradient_flow: u8 = @intFromBool(metadata.use_normalized_gradient_flow);
        const saved_embedding_seed = metadata.embedding_seed;
        const saved_default_max_seq_len = std.math.cast(usize, metadata.default_max_seq_len) orelse return TrainerError.InvalidEnvironmentValue;
        const saved_reasoning_cycles = std.math.cast(usize, metadata.reasoning_cycles) orelse return TrainerError.InvalidOptimizerConfiguration;
        const saved_relational_pass_interval = std.math.cast(usize, metadata.relational_pass_interval) orelse return TrainerError.InvalidRelationalPassInterval;
        const saved_shuffle_target_control: u8 = @intFromBool(metadata.shuffle_target_control);
        const saved_target_source_frozen: u8 = @intFromBool(metadata.target_source_frozen);
        const saved_spectral_depth_compensation: u8 = @intFromBool(metadata.spectral_depth_compensation);
        const saved_shuffle_control_state = metadata.shuffle_control_state;
        const saved_relational_fast_mode: u8 = @intFromBool(metadata.relational_fast_mode);
        const clip_min_f32 = metadata.clip_min;
        const clip_max_f32 = metadata.clip_max;
        const saved_rsf_optimizer_step = metadata.rsf_optimizer_step;
        if (saved_model_dim != self.model_dim) return TrainerError.ModelDimMismatch;
        if (saved_num_layers != self.num_layers) return TrainerError.NumLayersMismatch;
        if (saved_vocab_size == 0 or saved_vocab_size > self.config.max_distributed_integer) return TrainerError.VocabSizeMismatch;
        if (saved_local_batch_size == 0 or saved_local_batch_size > self.config.max_local_batch_size) return TrainerError.InvalidBatchSize;
        try validateHyperparameters(saved_learning_rate, saved_momentum);
        if (!std.math.isFinite(saved_fisher_gamma) or saved_fisher_gamma < 0.0 or saved_fisher_gamma >= 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(saved_fisher_epsilon) or saved_fisher_epsilon < 1e-12) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(saved_trust_ratio) or saved_trust_ratio <= 0.0 or saved_trust_ratio > 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(saved_weight_floor) or saved_weight_floor <= 0.0) return TrainerError.InvalidOptimizerConfiguration;
        if (saved_spectral_interval == 0 or !std.math.isFinite(saved_spectral_target) or saved_spectral_target <= 0.0) return TrainerError.InvalidSpectralState;
        if (!std.math.isFinite(saved_reconstruction_alpha) or saved_reconstruction_alpha < 0.0 or saved_reconstruction_alpha > 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(saved_logdet_weight)) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(saved_gradient_clip_norm) or saved_gradient_clip_norm <= 0.0) return TrainerError.InvalidGradient;
        if (saved_default_max_seq_len == 0 or saved_default_max_seq_len > self.config.max_distributed_integer) return TrainerError.InvalidEnvironmentValue;
        if (saved_reasoning_cycles == 0) return TrainerError.InvalidOptimizerConfiguration;
        if (saved_relational_pass_interval == 0) return TrainerError.InvalidRelationalPassInterval;
        if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or clip_min_f32 >= clip_max_f32) return TrainerError.InvalidClipRange;
        const clip_min = try checkedF32ToF16(clip_min_f32);
        const clip_max = try checkedF32ToF16(clip_max_f32);
        const half = self.model_dim / 2;
        const columns = accel.rsf_coupling_width;
        const per_layer = try std.math.mul(usize, half, columns);
        const total_rsf_state = try std.math.mul(usize, per_layer, self.num_layers);
        const saved_master_weights_s = try self.readCheckpointF32Array(reader, total_rsf_state, false);
        defer self.allocator.free(saved_master_weights_s);
        const saved_master_weights_t = try self.readCheckpointF32Array(reader, total_rsf_state, false);
        defer self.allocator.free(saved_master_weights_t);
        const saved_momentum_s = try self.readCheckpointF32Array(reader, total_rsf_state, false);
        defer self.allocator.free(saved_momentum_s);
        const saved_momentum_t = try self.readCheckpointF32Array(reader, total_rsf_state, false);
        defer self.allocator.free(saved_momentum_t);
        const saved_fisher_s = try self.readCheckpointF32Array(reader, total_rsf_state, true);
        defer self.allocator.free(saved_fisher_s);
        const saved_fisher_t = try self.readCheckpointF32Array(reader, total_rsf_state, true);
        defer self.allocator.free(saved_fisher_t);
        const has_embedding = try reader.readByte();
        if (has_embedding > 1) return TrainerError.InvalidCheckpointEmbeddingFlag;
        var pending_emb_master: ?[]f32 = null;
        var pending_emb_momentum: ?[]f32 = null;
        var pending_emb_fisher: ?[]f32 = null;
        var pending_emb_step: u64 = 0;
        var pending_emb_vocab: usize = 0;
        var pending_emb_dim: usize = 0;
        defer if (pending_emb_master) |values| self.allocator.free(values);
        defer if (pending_emb_momentum) |values| self.allocator.free(values);
        defer if (pending_emb_fisher) |values| self.allocator.free(values);
        if (has_embedding == 1) {
            pending_emb_vocab = std.math.cast(usize, try reader.readInt(u64, .little)) orelse return TrainerError.VocabSizeMismatch;
            pending_emb_dim = std.math.cast(usize, try reader.readInt(u64, .little)) orelse return TrainerError.ModelDimMismatch;
            if (pending_emb_vocab != saved_vocab_size) return TrainerError.VocabSizeMismatch;
            if (pending_emb_dim != self.model_dim) return TrainerError.ModelDimMismatch;
            const embedding_total = try std.math.mul(usize, pending_emb_vocab, pending_emb_dim);
            pending_emb_step = try reader.readInt(u64, .little);
            pending_emb_master = try self.readCheckpointF32Array(reader, embedding_total, false);
            pending_emb_momentum = try self.readCheckpointF32Array(reader, embedding_total, false);
            pending_emb_fisher = try self.readCheckpointF32Array(reader, embedding_total, true);
        }
        const has_target_source = try reader.readByte();
        if (has_target_source > 1) return TrainerError.InvalidCheckpointEmbeddingFlag;
        if ((saved_target_source_frozen == 1) != (has_target_source == 1)) return TrainerError.InvalidCheckpointEmbeddingFlag;
        var pending_target_master: ?[]f32 = null;
        var pending_target_vocab: usize = 0;
        var pending_target_dim: usize = 0;
        defer if (pending_target_master) |values| self.allocator.free(values);
        if (has_target_source == 1) {
            pending_target_vocab = std.math.cast(usize, try reader.readInt(u64, .little)) orelse return TrainerError.VocabSizeMismatch;
            pending_target_dim = std.math.cast(usize, try reader.readInt(u64, .little)) orelse return TrainerError.ModelDimMismatch;
            if (pending_target_vocab != saved_vocab_size) return TrainerError.VocabSizeMismatch;
            if (pending_target_dim != self.model_dim) return TrainerError.ModelDimMismatch;
            const target_total = try std.math.mul(usize, pending_target_vocab, pending_target_dim);
            pending_target_master = try self.readCheckpointF32Array(reader, target_total, false);
        }
        var loaded_nonce: [32]u8 = undefined;
        try reader.readNoEof(loaded_nonce[0..]);
        const new_training_graph = try readNsirGraph(self.allocator, reader, self.config);
        var new_training_graph_committed = false;
        errdefer if (!new_training_graph_committed) {
            new_training_graph.deinit();
            self.allocator.destroy(new_training_graph);
        };
        const new_knowledge_graph = try readNsirGraph(self.allocator, reader, self.config);
        var new_knowledge_graph_committed = false;
        errdefer if (!new_knowledge_graph_committed) {
            new_knowledge_graph.deinit();
            self.allocator.destroy(new_knowledge_graph);
        };
        const tokenizer_length_u64 = try reader.readInt(u64, .little);
        const tokenizer_length = std.math.cast(usize, tokenizer_length_u64) orelse return TrainerError.InvalidTokenizerData;
        if (tokenizer_length == 0 or tokenizer_length > self.config.max_tokenizer_file_size) return TrainerError.InvalidTokenizerData;
        const tokenizer_data = try self.allocator.alloc(u8, tokenizer_length);
        defer self.allocator.free(tokenizer_data);
        try reader.readNoEof(tokenizer_data);
        const trailer = try reader.readInt(u32, .little);
        if (trailer != CHECKPOINT_TRAILER) return TrainerError.CheckpointCorrupted;
        if (fbs.pos != checkpoint_payload.len) return TrainerError.TrailingCheckpointData;
        const tokenizer_tmp = try self.makeTmpFilePath("tokenizer");
        defer self.allocator.free(tokenizer_tmp);
        var tokenizer_tmp_committed = false;
        defer if (!tokenizer_tmp_committed) deletePath(tokenizer_tmp);
        {
            const tokenizer_file = try createWriteFile(tokenizer_tmp);
            var closed = false;
            defer if (!closed) tokenizer_file.close();
            try tokenizer_file.writer().writeAll(tokenizer_data);
            try tokenizer_file.sync();
            tokenizer_file.close();
            closed = true;
        }
        var new_tokenizer = try createConfiguredTokenizer(self.allocator, self.config);
        var new_tokenizer_committed = false;
        errdefer if (!new_tokenizer_committed) new_tokenizer.deinit();
        try new_tokenizer.loadVocab(tokenizer_tmp);
        if (new_tokenizer.next_token_id != saved_vocab_size) return TrainerError.VocabSizeMismatch;
        if (pending_emb_master != null and pending_emb_vocab != new_tokenizer.next_token_id) return TrainerError.VocabSizeMismatch;
        if (pending_target_master != null and pending_target_vocab != new_tokenizer.next_token_id) return TrainerError.VocabSizeMismatch;
        try admitEstimate(gpu_memory.EstimateConfig{
            .model_dim = @intCast(self.model_dim),
            .num_layers = @intCast(self.num_layers),
            .vocab_size = @intCast(saved_vocab_size),
            .batch_size = @intCast(self.local_batch_size),
            .max_seq_len = @intCast(self.runtime_max_seq_len),
            .world_size = self.coordinator.world_size,
            .stack_only = true,
            .frozen_target_f16_only = saved_target_source_frozen == 1,
            .include_graph_chunk = false,
            .spectral_temps = false,
            .coupling = .diagonal,
        });
        var new_accelerator_ptr = try self.allocator.create(RSFAccelerator);
        var new_accelerator_ptr_committed = false;
        errdefer if (!new_accelerator_ptr_committed) self.allocator.destroy(new_accelerator_ptr);
        new_accelerator_ptr.* = try RSFAccelerator.initMultiLayerWithDepthScale(
            self.model_dim,
            self.num_layers,
            self.allocator,
            self.config.spectral_depth_compensation,
        );
        var new_accelerator_committed = false;
        errdefer if (!new_accelerator_committed) new_accelerator_ptr.deinit();
        try new_accelerator_ptr.setOptimizerState(
            saved_master_weights_s,
            saved_master_weights_t,
            saved_momentum_s,
            saved_momentum_t,
            saved_fisher_s,
            saved_fisher_t,
            saved_rsf_optimizer_step,
        );
        try new_accelerator_ptr.setClipRange(clip_min, clip_max);
        try new_accelerator_ptr.sync();
        var loaded_gpu_embedding: ?accel.EmbeddingAccelerator = null;
        var loaded_gpu_embedding_committed = false;
        errdefer if (!loaded_gpu_embedding_committed) {
            if (loaded_gpu_embedding) |*emb| emb.deinit();
        };
        if (pending_emb_master) |master| {
            loaded_gpu_embedding = try accel.EmbeddingAccelerator.initWithMasterWeights(
                &new_accelerator_ptr.ctx,
                self.allocator,
                pending_emb_vocab,
                pending_emb_dim,
                master,
            );
            try loaded_gpu_embedding.?.setOptimizerState(
                master,
                pending_emb_momentum orelse return TrainerError.CheckpointCorrupted,
                pending_emb_fisher orelse return TrainerError.CheckpointCorrupted,
                pending_emb_step,
            );
        }
        var loaded_target_source: ?accel.FrozenEmbedding = null;
        var loaded_target_source_committed = false;
        errdefer if (!loaded_target_source_committed) {
            if (loaded_target_source) |*source| source.deinit();
        };
        if (pending_target_master) |master| {
            loaded_target_source = try accel.FrozenEmbedding.initFromMasterWeights(
                &new_accelerator_ptr.ctx,
                self.allocator,
                pending_target_vocab,
                pending_target_dim,
                master,
            );
        }
        try self.coordinator.synchronize();
        if (self.gpu_embedding) |*old_emb| old_emb.deinit();
        self.gpu_embedding = loaded_gpu_embedding;
        loaded_gpu_embedding_committed = true;
        if (self.target_source) |*old_source| old_source.deinit();
        self.target_source = loaded_target_source;
        loaded_target_source_committed = true;
        self.resetSpectralState();
        self.accelerator.deinit();
        self.allocator.destroy(self.accelerator);
        self.accelerator = new_accelerator_ptr;
        new_accelerator_ptr_committed = true;
        new_accelerator_committed = true;
        self.nsir_graph.deinit();
        self.allocator.destroy(self.nsir_graph);
        self.nsir_graph = new_training_graph;
        new_training_graph_committed = true;
        self.knowledge_nsir_graph.deinit();
        self.allocator.destroy(self.knowledge_nsir_graph);
        self.knowledge_nsir_graph = new_knowledge_graph;
        new_knowledge_graph_committed = true;
        self.tokenizer.deinit();
        self.tokenizer = new_tokenizer;
        new_tokenizer_committed = true;
        self.releaseKnowledgeFndsResources();
        self.vocab_size = saved_vocab_size;
        self.learning_rate = saved_learning_rate;
        self.momentum = saved_momentum;
        self.global_step = saved_global_step;
        self.config.learning_rate = saved_learning_rate;
        self.config.momentum = saved_momentum;
        self.config.fisher_gamma = saved_fisher_gamma;
        self.config.fisher_epsilon = saved_fisher_epsilon;
        self.config.trust_ratio = saved_trust_ratio;
        self.config.weight_floor = saved_weight_floor;
        self.config.optimizer_warmup_steps = saved_warmup_steps;
        self.config.spectral_interval = saved_spectral_interval;
        self.config.spectral_target_norm = saved_spectral_target;
        self.config.spectral_iterations = saved_spectral_iterations;
        self.config.reconstruction_alpha = saved_reconstruction_alpha;
        self.config.phase_a_steps = saved_phase_a_steps;
        self.config.phase_b_steps = saved_phase_b_steps;
        self.config.logdet_weight = saved_logdet_weight;
        self.config.gradient_clip_norm = saved_gradient_clip_norm;
        self.config.grad_mean = saved_grad_mean == 1;
        self.config.use_normalized_gradient_flow = saved_use_normalized_gradient_flow == 1;
        self.config.embedding_seed = saved_embedding_seed;
        self.config.reasoning_cycles = saved_reasoning_cycles;
        self.config.relational_pass_interval = saved_relational_pass_interval;
        self.config.shuffle_target_control = saved_shuffle_target_control == 1;
        self.config.target_source_frozen = saved_target_source_frozen == 1;
        self.config.spectral_depth_compensation = saved_spectral_depth_compensation == 1;
        self.shuffle_mutex.lock();
        self.shuffle_control_state = saved_shuffle_control_state;
        self.shuffle_mutex.unlock();
        self.relational_fast_mode = saved_relational_fast_mode == 1;
        self.spectral_normalizer = sfd.SpectralNormalizer.initWithConfig(.{
            .max_singular_value = saved_spectral_target,
        });
        self.knowledge_graph_nonce = loaded_nonce;
        self.last_step_telemetry = .{ .step = saved_global_step };
        self.step_synchronizer.mutex.lock();
        self.step_synchronizer.telemetry = self.last_step_telemetry;
        self.step_synchronizer.pending_step_increments = 0;
        self.step_synchronizer.mutex.unlock();
        tokenizer_tmp_committed = true;
        deletePath(tokenizer_tmp);
        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }
    fn ensureSpectralState(
        self: *DistributedTrainerFuthark,
        rows: usize,
        columns: usize,
    ) !void {
        if (self.gpu_spectral_u) |u| {
            if (self.gpu_spectral_v) |v| {
                if (u.len == rows and v.len == columns) return;
            }
        }
        self.resetSpectralState();
        const u_cpu = try self.allocator.alloc(f32, rows);
        defer self.allocator.free(u_cpu);
        const v_cpu = try self.allocator.alloc(f32, columns);
        defer self.allocator.free(v_cpu);
        const initial_u = 1.0 / @sqrt(@as(f32, @floatFromInt(rows)));
        const initial_v = 1.0 / @sqrt(@as(f32, @floatFromInt(columns)));
        @memset(u_cpu, initial_u);
        @memset(v_cpu, initial_v);
        const ctx = &self.accelerator.ctx;
        var new_u = try accel.FutharkArray1DF32.newFromSlice(ctx, u_cpu);
        errdefer new_u.free(ctx) catch |err| std.log.err("distributed trainer: releasing spectral u vector failed: {s}", .{@errorName(err)});
        const new_v = try accel.FutharkArray1DF32.newFromSlice(ctx, v_cpu);
        self.gpu_spectral_u = new_u;
        self.gpu_spectral_v = new_v;
    }
    fn applyEmbeddingSpectralNormalization(self: *DistributedTrainerFuthark) !void {
        if (self.gpu_embedding == null) return;
        const embedding = &self.gpu_embedding.?;
        const rows = embedding.vocab_size;
        const columns = embedding.dim;
        if (rows == 0 or columns == 0 or self.config.spectral_iterations == 0) return;
        try self.ensureSpectralState(rows, columns);
        const u = &self.gpu_spectral_u.?;
        const v = &self.gpu_spectral_v.?;
        {
            const context = &self.accelerator.ctx;
            context.mutex.lock();
            defer context.mutex.unlock();
            try embedding.spectralNormalize(
                u,
                v,
                self.config.spectral_iterations,
                self.spectral_normalizer.max_singular_value,
            );
        }
        try self.accelerator.sync();
    }
    pub fn buildKnowledgeGraph(self: *DistributedTrainerFuthark, text: []const u8) !void {
        if (text.len == 0) return TrainerError.EmptyKnowledgeGraphInput;
        if (text.len > self.config.max_knowledge_graph_input) return TrainerError.KnowledgeGraphInputTooLarge;
        return TrainerError.InvalidTrainingState;
    }
};
