def printResults(tuple_list: list, name_list: list):

    # Counter for street sign text
    street_sign_count = 0

    for (sign_type, present_flag, depth_val) in tuple_list:

        # Skip entry if sign is not present
        if not present_flag:
            continue

        # Process depth
        if (depth_val <= 0.25):
            depth_text = 'FAR AWAY'
        else:
            depth_text = 'CLOSE BY'

        # Process street sign
        if (sign_type == 'street sign'):
            
            # Get street sign
            sign_text = name_list[street_sign_count].upper()
            street_sign_count += 1

            # Print output
            print(f"Street \033[34m{sign_text}\033[0m detected \033[34m{depth_text}\033[0m")

        # Process all other signs
        else:

            # Get sign text
            sign_text = sign_type.upper()

            # Print output
            print(f"\033[34m{sign_text}\033[0m detected \033[34m{depth_text}\033[0m")