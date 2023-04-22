wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xQqKU0jMqiPCTUl_6yB-mxKqMFPco5zT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xQqKU0jMqiPCTUl_6yB-mxKqMFPco5zT" -O data_guacamol.zip && rm -rf /tmp/cookies.txt

mkdir -p ../data/guacamol

unzip data_guacamol.zip -d ../data/guacamol/

mv ../data/guacamol/data_guacamol/* ../data/guacamol/
mv ../data/guacamol/retrieval_database_guacamol ../data/guacamol/retrieval_database

rm -r ../data/guacamol/data_guacamol
rm -r ../data/guacamol/__MACOSX
rm data_guacamol.zip