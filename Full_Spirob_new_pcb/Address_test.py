def calculate_ltc4316_real():
    """
    Berechnet alle möglichen Adressen-Kombinationen für eine einzelne Platine
    mit MMA8452Q + MLX90393 + LTC4316 durch Umschalten von XORL/XORH.
    
    XORL: Untere 4 Bit (a3, a2, a1, a0) - 16 Kombinationen
    XORH: Obere 3 Bit (a6, a5, a4) - 8 Kombinationen
    Total: 128 mögliche Kombinationen
    """
    
    # Basis-Adressen deiner Sensoren
    sensor_1_internal = 0x1C  # MMA8452Q
    sensor_2_internal = 0x0C  # MLX90393
    
    # Reservierte I2C Bereiche (können nicht verwendet werden)
    reserved_range = set(list(range(0x00, 0x08)) + list(range(0x78, 0x80)))

    print("=" * 70)
    print("EINE PLATINE - ADRESSEN-KOMBINATIONEN DURCH PIN-UMSCHALTUNG")
    print("=" * 70 + "\n")
    
    print(f"Basis-Adressen (intern):")
    print(f"  MMA8452Q: 0x{sensor_1_internal:02X}")
    print(f"  MLX90393: 0x{sensor_2_internal:02X}\n")
    
    print(f"Pin-Konfiguration:")
    print(f"  XORL (Low): 4 Bit (a3, a2, a1, a0) -> 2^4 = 16 Kombinationen")
    print(f"  XORH (High): 3 Bit (a6, a5, a4) -> 2^3 = 8 Kombinationen")
    print(f"  Total: 16 x 8 = 128 moegliche Kombinationen\n")
    
    print(f"{'XORH':<6} | {'XORL':<6} | {'XOR-Byte':<10} | {'MMA':<8} | {'MLX':<8} | {'Status':<8}")
    print("-" * 70)

    valid_combinations = []
    
    # XORH: 3 Bit (obere) - 8 Kombinationen (0-7)
    # XORL: 4 Bit (untere) - 16 Kombinationen (0-15)
    for xorh in range(8):  # 3 Bit
        for xorl in range(16):  # 4 Bit
            # Kombiniere die Bits
            translation_byte = (xorh << 4) | xorl
            
            ext_addr_1 = (sensor_1_internal ^ translation_byte) & 0x7F
            ext_addr_2 = (sensor_2_internal ^ translation_byte) & 0x7F
            
            # Prüfe ob beide Adressen gültig sind (nicht reserviert)
            is_valid = (ext_addr_1 not in reserved_range) and (ext_addr_2 not in reserved_range)
            
            if is_valid:
                valid_combinations.append((xorh, xorl, translation_byte, ext_addr_1, ext_addr_2))
                status = "[OK]"
            else:
                status = "[X]"
            
            # Nur alle 2 Zeilen drucken um nicht zu viel Output
            if xorl % 2 == 0:
                print(f"{xorh:<6} | {xorl:<6} | 0x{translation_byte:02X}     | 0x{ext_addr_1:02X}   | 0x{ext_addr_2:02X}   | {status:<8}")

    print("\n" + "=" * 70)
    print("\nDATA ANALYSIS:\n")
    print(f"  Valid Pin-Combinations: {len(valid_combinations)} / 128")
    print(f"  These combinations have no reserved addresses!\n")
    
    # Check for address collisions between configurations
    print("ADDRESS COLLISION CHECK:\n")
    
    # Collect all addresses from all valid configurations
    all_addresses = {}  # address -> list of (config_idx, sensor_type)
    
    for i, (xh, xl, trans, addr1, addr2) in enumerate(valid_combinations):
        if addr1 not in all_addresses:
            all_addresses[addr1] = []
        all_addresses[addr1].append((i, xh, xl, "MMA"))
        
        if addr2 not in all_addresses:
            all_addresses[addr2] = []
        all_addresses[addr2].append((i, xh, xl, "MLX"))
    
    # Find duplicates
    duplicates = {addr: configs for addr, configs in all_addresses.items() if len(configs) > 1}
    
    if duplicates:
        print(f"  DUPLICATES FOUND: {len(duplicates)} addresses used multiple times!\n")
        print(f"  Examples (first 5):")
        for addr in sorted(duplicates.keys())[:5]:
            configs = duplicates[addr]
            print(f"  Address 0x{addr:02X} used in:")
            for cfg_idx, xh, xl, sensor_type in configs:
                print(f"    - Config {cfg_idx+1}: XORH={xh}, XORL={xl:2d} -> {sensor_type}")
    else:
        print(f"  NO DUPLICATES - All {len(all_addresses)} addresses unique!\n")
    
    # Calculate maximum number of boards
    print("\n" + "=" * 70)
    print("\nMAXIMUM BOARDS WITH DIFFERENT CONFIGURATIONS:\n")
    
    print("THEORETICAL ANALYSIS:")
    print(f"  Total unique addresses: {len(all_addresses)}")
    print(f"  Addresses per board: 2 (1x MMA8452Q + 1x MLX90393)")
    print(f"  Maximum boards = {len(all_addresses)} / 2 = {len(all_addresses) // 2}\n")
    
    # Get address sets for each configuration
    config_addresses = {}
    for i, (xh, xl, trans, addr1, addr2) in enumerate(valid_combinations):
        config_addresses[i] = {addr1, addr2}
    
    # Find maximum compatible set (greedy algorithm)
    used_addresses = set()
    selected_configs = []
    
    for config_idx, addrs in config_addresses.items():
        if not (addrs & used_addresses):  # No intersection = no collision
            selected_configs.append(config_idx)
            used_addresses.update(addrs)
    
    print("GREEDY ALGORITHM RESULT:")
    print(f"  Selected configurations: {len(selected_configs)}")
    print(f"  Total addresses used: {len(used_addresses)}")
    print(f"  Total addresses available: {len(all_addresses)}")
    print(f"  Utilization: {len(used_addresses)}/{len(all_addresses)} ({100*len(used_addresses)//len(all_addresses)}%)\n")
    
    print("=" * 70)
    print(f"\nRESULT: Du kannst {len(selected_configs)} Platinen gleichzeitig verwenden!\n")
    print("=" * 70)
    print(f"\nBEGRUENDUNG:")
    print(f"  - Jede Platine belegt 2 I2C-Adressen (MMA8452Q + MLX90393)")
    print(f"  - Insgesamt gibt es {len(all_addresses)} eindeutige Adressen")
    print(f"  - {len(all_addresses)} Adressen / 2 Sensoren = {len(selected_configs)} Platinen")
    print(f"  - Alle {len(selected_configs)} Platinen haben unterschiedliche Adressen\n")
    
    print("Optimale Konfiguration ({} boards, NO address conflicts):".format(len(selected_configs)))
    print("-" * 70)
    for idx, config_idx in enumerate(selected_configs, 1):
        xh, xl, trans, addr1, addr2 = valid_combinations[config_idx]
        print(f"  Board {idx:2d}: XORH={xh}, XORL={xl:2d} -> MMA=0x{addr1:02X}, MLX=0x{addr2:02X}")

calculate_ltc4316_real()