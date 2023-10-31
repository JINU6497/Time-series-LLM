for script in scripts/TMAE2/*.sh; do
    chmod +x "$script"     # Make the script executable if it's not already
    ./"$script"            # Execute the script
done