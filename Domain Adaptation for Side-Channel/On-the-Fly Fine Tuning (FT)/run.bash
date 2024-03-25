# Define the range for XXXX values
for (( XXXX = 10000; XXXX <= 10000; XXXX += 3000 )); do
    # Define the range for YY values
    for YY in 12 ; do
        # Run first Python command
        python OTF_step_cuda.py \
        -i /home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/RandomDelay/og//X2_K2_150k_L${YY}_delay_20.npz \
        -o "/home/mabon/Cross_EM/FINETUNE/MUTLIRUNNER/xmega_em/T1/randomdelay/onthefly/L$YY/$XXXX/" \
        -tb 2 \
        -aw 1800_2800 \
        -tn $((XXXX*2))\
        -ts $XXXX \
        -lm HW \
        -e 150 \
        -m "/home/mabon/Cross_EM/model/CNN/TIME/HW/xmega_em/T1/RandomDelay/L11"\
        -ftl 2\
        -v
    done
done
