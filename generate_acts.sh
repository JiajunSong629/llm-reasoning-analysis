# python generate_acts.py --model llama-2-7b --layers 14 \
#     --datasets cities neg_cities cities_alice neg_cities_alice \
#     --device cuda:0

# python generate_acts.py --model llama-3-8b --layers 8 10 12 \
#     --datasets cities neg_cities cities_alice neg_cities_alice \
#     --device cuda:0

python generate_acts.py --model llama-3-8b --layers 8 10 12 14\
    --datasets xor_letters \
    --device cuda:0