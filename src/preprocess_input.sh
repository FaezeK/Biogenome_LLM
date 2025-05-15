# preprocess each chromosome

# remove header
grep -v '>' chr_original_file.fa > chr_no_header.fa

# concat all segments in the original FASTA file
tr -d '\n' < chr_no_header.fa > chr_one_line.txt

# remove extra nucleotides to have length divisible by 1000
var=$(cat chr_one_line.txt); size=${#var}; echo ${var:0:$(($((size/1000))*1000))} > chr_divisible_by_1000.txt

# breakdown the one liner files to multiple lines of length 1000 with half of lines overlapping
var=$(cat chr_divisible_by_1000.txt); size=${#var}; input_len=1000; num_lines=$((2 * $((size / $input_len)))); for i in $(seq 1 $(($num_lines-1))); do start=$(($(($input_len / 2)) * $(($i-1)))); echo ${var:start:input_len}; done > chr_1000.txt

# get reverse complement strands
while read l; do echo $l | tr 'ATCGatcg' 'TAGCtagc' | rev; done < chr_1000.txt > chr_1000_rev.txt

# concat original sequence and reverse complement into one file
cat chr_1000.txt chr_1000_rev.txt > chr_orig_n_reverse_1000.csv
