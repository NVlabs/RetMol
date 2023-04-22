wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nN1KEDPNuANJqZJwAOPMKwdHxMa6J7F3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nN1KEDPNuANJqZJwAOPMKwdHxMa6J7F3" -O data-retrieval-precompute.zip && rm -rf /tmp/cookies.txt

unzip data-retrieval-precompute.zip -d ../data/

rm data-retrieval-precompute.zip