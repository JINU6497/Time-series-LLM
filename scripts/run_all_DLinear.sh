for script in scripts/DLinear/*.sh; do
    chmod +x "$script"     # Make the script executable if it's not already
    ./"$script"            # Execute the script
done