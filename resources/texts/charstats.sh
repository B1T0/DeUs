char=$1  
echo $char >> charstats.txt
echo `fgrep -o $char goethe.txt | wc -l`,`wc -m goethe.txt` >> charstats.txt
echo `fgrep -o $char kafka.txt | wc -l`,`wc -m kafka.txt` >> charstats.txt
echo `fgrep -o $char kleist.txt | wc -l`,`wc -m kleist.txt` >> charstats.txt
echo `fgrep -o $char raabe.txt | wc -l`,`wc -m raabe.txt` >> charstats.txt
echo `fgrep -o $char schiller.txt | wc -l`,`wc -m schiller.txt` >> charstats.txt

