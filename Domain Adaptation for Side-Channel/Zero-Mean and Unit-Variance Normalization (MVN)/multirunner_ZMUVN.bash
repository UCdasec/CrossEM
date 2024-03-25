for (( XXXX =50 ; XXXX>= 10; XXXX -= 10 )); do
    # Define the range for YY values
    for YY in 01 02 10 11 20 22; do

        # Run first Python command
        python ZMUVN.py \
        -i "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T1/RandomDelay/ew/X1_K1_200k_L11_delay_20.npz" \
        -t "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/RandomDelay/og/X2_K2_150k_L${YY}_delay_20.npz" \
        -tnt $XXXX \
        -o "/home/mabon/Cross_EM/ZMUVN/datasets/xmega_em/T1/RandomDelay/L11_L$YY/$XXXX/"
        
        # Run second Python command
        python cnn/test.py \
        -i "/home/mabon/Cross_EM/ZMUVN/datasets/xmega_em/T1/RandomDelay/L11_L$YY/$XXXX/Test.npz" \
        -m "/home/mabon/Cross_EM/model/CNN/TIME/ID/xmega_em/T1/RandomDelay/L11/" \
        -o "/home/mabon/Cross_EM/ZMUVN/cnn_output/xmega_em/T1/RandomDelay/L11_L$YY/$XXXX/" \
        -tb 2 \
        -aw 1800_2800 \
        -tn $XXXX \
        -lm ID
    done
done

for (( XXXX =50 ; XXXX>= 10; XXXX -= 10 )); do
    # Define the range for YY values
    for YY in 11 ; do

        # Run first Python command
        python ZMUVN.py \
        -i "/home/mabon/Cross_EM/datasets/TIME/stm_em/T1/RandomDelay/S1_K1_150k_L11_delay_20.npz" \
        -t "/home/mabon/Cross_EM/datasets/TIME/stm_em/T2/RandomDelay/og/S2_K3_150k_L${YY}_delay_20.npz" \
        -tnt $XXXX \
        -o "/home/mabon/Cross_EM/ZMUVN/datasets/stm_em/T1/RandomDelay/L11_L$YY/$XXXX/"
        
        # Run second Python command
        python cnn/test.py \
        -i "/home/mabon/Cross_EM/ZMUVN/datasets/stm_em/T1/RandomDelay/L11_L$YY/$XXXX/Test.npz" \
        -m "/home/mabon/Cross_EM/model/CNN/TIME/ID/stm_em/T1/RandomDelay/" \
        -o "/home/mabon/Cross_EM/ZMUVN/cnn_output/stm_em/T1/RandomDelay/L11_L$YY/$XXXX/" \
        -tb 2 \
        -aw 1200_2200 \
        -tn $XXXX \
        -lm ID
    done
done



# # # Define the range for XXXX values
# for (( XXXX = 4000 ; XXXX>= 500; XXXX -= 500 )); do
#     # Define the range for YY values
#     for YY in 01 02 10 11 12 20 21 ; do

#         # Run first Python command
#         python ZMUVN.py \
#         -i "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/X2_K2_150k_L11.npz" \
#         -t "/home/mabon/Cross_EM/datasets/TIME/xmega_em/T2/X2_K2_150k_L${YY}.npz" \
#         -tnt $XXXX \
#         -o "/home/mabon/Cross_EM/ZMUVN/datasets/xmega_em/T2/L11_L$YY/$XXXX/"
        
#         # Run second Python command
#         python cnn/test.py \
#         -i "/home/mabon/Cross_EM/ZMUVN/datasets/xmega_em/T2/L11_L$YY/$XXXX/Test.npz" \
#         -m "/home/mabon/Cross_EM/model/CNN/TIME/ID/xmega_em/T2/L11/" \
#         -o "/home/mabon/Cross_EM/ZMUVN/cnn_output/xmega_em/T2/L11_L$YY/$XXXX/" \
#         -tb 2 \
#         -aw 1800_2800 \
#         -tn $XXXX \
#         -lm ID
#     done
# done

# # # Define the range for XXXX values
# for (( XXXX = 400 ; XXXX>= 50; XXXX -= 50 )); do
#     # Define the range for YY values
#     for YY in 01 10 11 12 ; do
#        # Run first Python command
#         python ZMUVN.py \
#         -i "/home/mabon/Cross_EM/datasets/TIME/stm_em/T2/S2_K3_150k_L11.npz" \
#         -t "/home/mabon/Cross_EM/datasets/TIME/stm_em/T2/S2_K3_150k_L${YY}.npz" \
#         -tnt $XXXX \
#         -o "/home/mabon/Cross_EM/ZMUVN/datasets/stm_em/T2/L11_L$YY/$XXXX/"
        
#         # Run second Python command
#         python cnn/test.py \
#         -i "/home/mabon/Cross_EM/ZMUVN/datasets/stm_em/T2/L11_L$YY/$XXXX/Test.npz" \
#         -m "/home/mabon/Cross_EM/model/CNN/TIME/ID/stm_em/T2/L11/" \
#         -o "/home/mabon/Cross_EM/ZMUVN/cnn_output/stm_em/T2/L11_L$YY/$XXXX/" \
#         -tb 2 \
#         -aw 1200_2200 \
#         -tn $XXXX \
#         -lm ID
#     done
# done