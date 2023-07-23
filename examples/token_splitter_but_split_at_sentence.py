# This splitter will split by tokens. But will ensure that the split does not happen
# at the middle of the sentence. So, some splits can be greater than the chunk_split
import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = """
I could be a social media manager, I could be a blog writer, or I could add a little bowstring to my bow and become a scientific writer or work a writer in biotech and you'll get paid.
I should be eating healthier, but it's really hard for me.
Another person is going to say, gosh, I know I should be eating healthier, but it's really hard for me.
"""

tokens = enc.encode(text)
print(tokens)
split = []
splits =[]
chunk_size = 50
for token in tokens:
    if token == 13:
        split.append(13)
        splits.append(split)
        split=[]
    else:
        split.append(token)

print(splits)
merged_split=""
merged_splits=[]
merged_lenth=0
for split in splits:
    merged_lenth = merged_lenth + len(split)
    if merged_lenth <= chunk_size:
        merged_split = merged_split + enc.decode(split)
        print(merged_split)
        print(merged_lenth)
    else:
        merged_split = merged_split + enc.decode(split)
        merged_splits.append(merged_split)
        merged_split=""
        merged_lenth=0
merged_splits.append(merged_split)

for merge_split in merged_splits:
    print("---")
    print(merge_split)
    print(len(enc.encode(merge_split)))
