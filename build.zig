const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const opts = .{ .target = target, .optimize = optimize };
    const json_module = b.dependency("json", opts).module("json");

    // we name the module duck which will be used later
    const main_module = b.addModule("zigNEAT", .{
        .source_file = .{ .path = "src/zigNEAT.zig" },
        .dependencies = &.{
            .{ .name = "json", .module = json_module },
        },
    });

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const main_tests = b.addTest(.{
        .root_source_file = main_module.source_file,
        .target = target,
        .optimize = optimize,
    });

    main_tests.addModule("json", json_module);

    const run_main_tests = b.addRunArtifact(main_tests);

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build test`
    // This will evaluate the `test` step rather than the default, which is "install".
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);

    // Docs
    const zigNEAT_docs = main_tests;
    const build_docs = b.addInstallDirectory(.{
        .source_dir = zigNEAT_docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "../docs",
    });
    const build_docs_step = b.step("docs", "Build the zigNEAT library docs");
    build_docs_step.dependOn(&build_docs.step);
}
