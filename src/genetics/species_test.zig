test "Species adjust fitness" {
    var allocator = std.testing.allocator;
    var sp = try build_test_species_with_organisms(allocator, 1);
    defer sp.deinit();
    // configuration
    var options = Options{
        .dropoff_age = 5,
        .survival_thresh = 0.5,
        .age_significance = 0.5,
    };
    sp.adjust_fitness(&options);

    // verify results
    try std.testing.expect(sp.organisms.items[0].is_champion);
    try std.testing.expect(sp.age_of_last_improvement == 1);
    try std.testing.expect(sp.max_fitness_ever == 15.0);
    try std.testing.expect(sp.organisms.items[2].to_eliminate);
}
