
# # Define the range for XXXX values
for (( XXXX = 10 ; XXXX>= 1; XXXX -= 1 )); do
    # Define the range for YY values
    for YY in  01 02 10; do

        # Run first Python command
        python Domain.py \
        -i "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T1/RandomDelay/ew/X1_K1_200k_L11_delay_20.npz" \
        -t "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/RandomDelay/og/X2_K2_150k_L${YY}_delay_20.npz" \
        -tnt $XXXX \
        -o "/home/mabon/Cross_EM/Update_DA/xmega_em/multirunner2/T1/RandomDelay/L11_L$YY/$XXXX/"
        
        # Run second Python command
        python cnn/test.py \
        -i "/home/mabon/Cross_EM/Update_DA/xmega_em/multirunner2/T1/RandomDelay/L11_L$YY/$XXXX/Test.npz" \
        -m "/home/mabon/Cross_EM/model/CNN/TIME/ID/xmega_em/T1/RandomDelay/L11/" \
        -o "/home/mabon/Cross_EM/Update_DA/xmega_em/multirunner2/T1/RandomDelay/L11_L$YY/$XXXX/" \
        -tb 2 \
        -aw 1800_2800 \
        -tn $XXXX \
        -lm ID
    done
done


# # # Define the range for XXXX values
# for (( XXXX = 500; XXXX<= 8000; XXXX += 500 )); do
#     # Define the range for YY values
# #    for YY in 00 01 02 10 11 12 20 21 22; do
#     for YY in 20; do

#         # Run first Python command
#         python Domain.py \
#         -i "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/X2_K2_150k_L11.npz" \
#         -t "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/X2_K2_150k_L$YY.npz" \
#         -tnt $XXXX \
#         -o "/home/mabon/Cross_EM/Update_DA/xmega_em/multirunner2/T2/L11_L$YY/$XXXX/"
        
#         # Run second Python command
#         python cnn/test.py \
#         -i "/home/mabon/Cross_EM/Update_DA/xmega_em/multirunner2/T2/L11_L$YY/$XXXX/Test.npz" \
#         -m "/home/mabon/Cross_EM/model/CNN/TIME/ID/xmega_em/T2/L11/" \
#         -o "/home/mabon/Cross_EM/Update_DA/xmega_em/multirunner2/T2/L11_L$YY/$XXXX/" \
#         -tb 2 \
#         -aw 1800_2800 \
#         -tn $XXXX \
#         -lm ID
#     done
# done
