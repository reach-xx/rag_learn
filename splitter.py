
# 把文档分割成知识块
import jieba, re

def split_text_by_sentences(source_text: str,
                            sentences_per_chunk: int,
                            overlap: int) -> list[str]:
    """
    简单地把文档分割为多个知识块，每个知识块都包含指定数量的句子
    """
    if sentences_per_chunk < 2:
        raise ValueError("一个句子至少要有2个chunk！")
    if overlap < 0 or overlap >= sentences_per_chunk - 1:
        raise ValueError("overlap参数必须大于等于0，且小于sentences_per_chunk")

    # 简单化处理，用正则表达式分割句子
    sentences = re.split('(?<=[。！？])\s+', source_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']

    if not sentences:
        print("Nothing to chunk")
        return []

        # 处理overlap参数
    chunks = []
    i = 0
    while i < len(sentences):

        end = min(i + sentences_per_chunk, len(sentences))
        chunk = ' '.join(sentences[i:end])

        if overlap > 0 and i > 1:
            overlap_start = max(0, i - overlap)
            overlap_end = i
            overlap_chunk = ' '.join(sentences[overlap_start:overlap_end])
            chunk = overlap_chunk + ' ' + chunk

        chunks.append(chunk.strip())
        i += sentences_per_chunk

    return chunks
