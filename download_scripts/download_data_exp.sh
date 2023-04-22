wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Si5_yHdCGZNHQov99hPp8rOZx4DC6BY_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Si5_yHdCGZNHQov99hPp8rOZx4DC6BY_" -O data_exp.zip && rm -rf /tmp/cookies.txt

mkdir -p ../data

unzip data_exp.zip -d ../data/

rm -r ../data/__MACOSX
rm data_exp.zip