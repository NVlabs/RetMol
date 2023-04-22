wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J5KVdJc7SMPCSF0Y8feoYH1DyHI-YF7B' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J5KVdJc7SMPCSF0Y8feoYH1DyHI-YF7B" -O data_cov.zip && rm -rf /tmp/cookies.txt

mkdir -p ../data

unzip data_cov.zip -d ../data/cov

mv ../data/cov/data_cov/* ../data/cov/
rm -r ../data/cov/data_cov
rm data_cov.zip
