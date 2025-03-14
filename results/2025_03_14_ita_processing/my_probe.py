channel_groups = {
    0: {
        'channels': list(range(16)),  # 16 channels based on provided data
        'geometry': {
            0: (-1.5, 1.5),  # Channel 15
            1: (-0.5, 1.5),  # Channel 19
            2: (0.5, 1.5),   # Channel 8
            3: (1.5, 1.5),   # Channel 20
            4: (-1.5, 0.5),  # Channel 14
            5: (-0.5, 0.5),  # Channel 18
            6: (0.5, 0.5),   # Channel 9
            7: (1.5, 0.5),   # Channel 21
            8: (-1.5, -0.5), # Channel 13
            9: (-0.5, -0.5), # Channel 17
            10: (0.5, -0.5), # Channel 10
            11: (1.5, -0.5), # Channel 22
            12: (-1.5, -1.5),# Channel 12
            13: (-0.5, -1.5),# Channel 16
            14: (0.5, -1.5), # Channel 11
            15: (1.5, -1.5), # Channel 23
        },
        'graph': [
            (0, 1), (1, 2), (2, 3),  # Top row
            (4, 5), (5, 6), (6, 7),  # Second row
            (8, 9), (9, 10), (10, 11),  # Third row
            (12, 13), (13, 14), (14, 15),  # Bottom row
            (0, 4), (4, 8), (8, 12),  # First column
            (1, 5), (5, 9), (9, 13),  # Second column
            (2, 6), (6, 10), (10, 14),  # Third column
            (3, 7), (7, 11), (11, 15),  # Fourth column
        ]
    }
}
