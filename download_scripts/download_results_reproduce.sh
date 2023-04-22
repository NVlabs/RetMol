wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16USnJttlMES1uPtRjJ7WNJ3UcXoTUhV1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16USnJttlMES1uPtRjJ7WNJ3UcXoTUhV1" -O results_reproduce.zip && rm -rf /tmp/cookies.txt

unzip results_reproduce.zip -d ../results_reproduce

rm results_reproduce.zip