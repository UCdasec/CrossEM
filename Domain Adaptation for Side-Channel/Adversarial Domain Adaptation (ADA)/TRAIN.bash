
# Define the range for XXXX values
for (( XXXX = 500; XXXX >= 500; XXXX -= 100 )); do
    # Define the range for YY values
    for YY in 10; do
        # Run first Python command
        python ada/revGrad.py \
        -s /workspace/datasets/TIME/xmega_em/T1/X1_K1_200k_L11_delay_20.npz \
        -t "/workspace/datasets/TIME/xmega_em/T2/og/X2_K2_150k_L${YY}_delay_20.npz" \
        -tb 2 \
        -saw 1800_2800 \
        -taw 1800_2800 \
        -srn 140000 \
        -trn $XXXX \
        -lm ID \
        -e 40 \
        -o "/workspace/ada/xmega_em/multirunner2/T1/RD/L11_L$YY/$XXXX"
        
        # Run second Python command
        python ada/ada_test.py \
        -i "/workspace/datasets/TIME/xmega_em/T2/og/X2_K2_150k_L${YY}_delay_20.npz" \
        -o "/workspace/ada/xmega_em/multirunner2/T1/RD/L11_L$YY/$XXXX" \
        -tb 2 \
        -aw 1800_2800 \
        -tn $XXXX \
        -lm ID
    done
done

