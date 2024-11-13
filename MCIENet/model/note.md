# CNN model
## what is 1D CNN 
https://www.kaggle.com/code/mersico/understanding-1d-2d-and-3d-convolution-network

## final FC layer shape 計算方式
輸出大小 = （輸入大小 - Kernel size + 2 * Padding）/ Stride + 1
"Input size" 表示輸入圖像或特徵映射的大小。
"Kernel size" 是卷積過濾器的大小。
"Padding" 是在輸入周圍添加的零值像素，以控制輸出大小。
"Stride" 是卷積過程中過濾器每次移動的距離。

conv1: ((6000 - 8 + 2 *0) / 1) + 1 = 5993
MaxPool1d: ((5993 - 4 + 2 * 0) / 4) + 1 = 1498

((1498 - 8) / 1) + 1 = 1491.0
((1491 - 4) / 4) + 1

## DeepSeaModel (use by Chinn)
### [model Args]
conv1: in(4), out(128), kernel_size(8), stride(1), padding(0)
conv2: in(128), out(256), kernel_size(8), stride(1), padding(0)
conv3: in(256), out(128), kernel_size(8), stride(1), padding(0)
maxpool(): kernel_size(4), stride(4), padding(0)

### [last layer]
最後一層有三種 method: 
  1. weightsum -> random 128*53 的參數後 kaiming_uniform(He initialization)初始化， 
      和原本的最後一層(batch_size, 128, 53)相乘，
      再對 dim 2 做 sum。
  2. flatten(128*53) + Linear*2
  3. 多一層 CNN + squeeze(移除 dim 為 1 的維度)  

### [Data shape]
conv1: [batch_size, 128, 993]
maxpool: [batch_size, 128, 248]
conv2: [batch_size, 256, 241]
maxpool: [batch_size, 256, 60]
conv3: [batch_size, 128, 53]

output [batch_size, 128]

## 


<!-- ========================================================== -->

# Transformers

## Encoder & Decoder 架構
- Encoder-Decoder 結構:
    Seq-to-Seq 任務: 例如機器翻譯、文本摘要等，其中模型需要將一個序列轉換為另一個序列。
    圖片到文本生成: 例如圖片標註，模型需要理解圖片並生成相應的文本描述。

- Encoder-Only 結構:
    文本分類: 如果你的任務是將一個文本序列分類為不同的類別，你可能只需要使用 Encoder 部分。
    語言建模: 對於單向語言模型，你可能只需使用 Encoder 部分來預測下一個詞。

- Decoder-Only 結構:
    生成式任務: 如果你的任務是生成一個序列，但不需要將另一個序列作為輸入（如語言生成）。
    條件生成: 如果你已經有一個上下文，並且希望生成與該上下文相關的序列。

簡而言之，Encoder-Decoder 結構通常適用於需要將一個序列轉換為另一個序列的任務。Encoder-Only 結構通常用於需要從輸入序列中提取信息並執行分類的任務。Decoder-Only 結構通常用於生成式任務，其中模型需要生成與某種上下文相關的序列。


## 在程式中的 source Sequence & target Sequence
- Encoder-Decoder 結構:
    - Source Sequence（源序列）：這是輸入到模型的原始數據序列，例如一段文本、一幅圖片，或其他任務相應的輸入。Encoder 被用來將源序列轉換為一系列的特徵表示。
    - Target Sequence（目標序列）：這是模型訓練時的目標輸出序列。在機器翻譯任務中，這可能是翻譯成的目標語言的句子；在文本摘要任務中，這可能是摘要的文本。Decoder 被用來生成這個目標序列。
- Encoder-Only 或 Decoder-Only 架構：
  - Source Sequence（源序列）：在 Encoder-Only 模型中，這是輸入到模型的原始數據序列，而在 Decoder-Only 模型中，這可能是一些上下文信息或先前生成的序列。
  - 在這些情況下，通常不存在一個目標序列。Encoder-Only 模型中，模型的目標可能是對源序列進行某種分類或特徵提取。在 Decoder-Only 模型中，目標通常是生成新的序列，例如語言生成或圖像生成。


## 關於 target data leakage 
對於一些特殊的任務，特別是生成式任務，Encoder-Decoder 模型的輸入確實可能包含了部分或全部的答案。這種情況在一些 seq2seq 任務中是很常見的，例如機器翻譯或文本摘要。

在這些情境下，模型必須學會從完整的輸入序列中提取有用的信息，並將這些信息轉換為目標序列。這也是 Transformer 模型的一個優勢，因為 Transformer 具有自注意機制，能夠捕捉輸入序列中的長距離相依性，使其能夠更好地處理這樣的情況。

然而，這種設計也帶來了一些挑戰，例如模型可能學會直接複製輸入序列的一部分或全部作為輸出，而不真正理解任務。這被稱為 "exposure bias" 或 "copying mechanism" 問題，並且需要額外的注意力來解決。

總體而言，這是一個權衡，具體應用中需要根據任務的特性和要求來調整模型的結構和設計。

為了解決這個問題，可以考慮以下方法：

- shifted right (原始的 Transformers)
    - 是的，"shifted right" 或稱為右移（right shift）操作通常用於處理 Seq2Seq（序列到序列）任務中的問題。這種右移的操作在訓練過程中用於確保在解碼（Decoder）階段，模型在生成每個詞時都是基於真實的前一個詞而不是生成的詞。
    - 在 Seq2Seq 任務中，通常會使用 "teacher forcing" 技術，即將真實的目標序列的詞作為解碼器的輸入，而不是使用模型自己生成的詞。這可以提高訓練穩定性，但也引入了一個問題，即在推理（測試）時，當模型沒有真實目標序列時，它如何生成序列。
    - 右移操作解決了這個問題。在解碼階段，每個時間步的輸入都是上一步生成的詞，即模型生成的詞。但是在計算損失（Loss）時，使用的目標序列是真實的目標序列右移一個位置，以便模型在每一步預測時都可以與實際的下一個詞進行比較。
    - 這種方式確保模型在訓練過程中能夠正確地學習到生成序列的連貫性，而不是依賴於已生成的序列的信息。這有助於提高模型的泛化能力並減少訓練時的資料洩漏問題。

- Teacher Forcing with Scheduled Sampling: 在訓練過程中，不總是使用真實的目標序列作為 Decoder 的輸入，而是以一定的概率使用模型自己生成的序列。這稱為 "scheduled sampling"，有助於減輕資料洩漏的問題。

- Masked Language Modeling Loss: 在一些生成式任務中，可以使用遮罩語言建模（masked language modeling）損失，即將部分輸入序列隨機遮蓋，並要求模型預測這些遮罩部分。這有助於使模型學會生成輸出而不依賴於特定的輸入。

- 在 Transformer 模型中，通常會使用一種被稱為 "greedy decoding" 的方法，即在每個時間步都選擇生成最有可能的單詞。或者，還可以使用束搜索（beam search）等解碼策略。

## sinusoidal embeddings 的必要性
embeddings 在 Transformer 架構中並非必要。Sinusoidal embeddings 是原始 Transformer 架構中使用的方法，它可以將單詞表示為連續的實數向量。這可以幫助模型更好地利用詞序信息。

- embeddings 的優缺點：
  - 優點: 可以幫助模型更好地利用詞序信息。可以提高模型的性能。
  - 缺點: 需要額外的參數。可能會增加模型的訓練時間。


## 參數調整
- d_model：隱藏表示的維度。
- n_head：注意力頭的數量。
- d_ff：前饋網路的隱藏層維度。
num_classes：輸出的類別數。
這些參數可以根據您的需要進行調整。

以下是一些調整參數的建議：

- d_model：d_model 的值越大，模型的表示能力就越強。但是，d_model 值過大也會增加模型的計算複雜度。一般來說d_model 的值可以從 512 到 2048 不等。
- n_head：n_head 的值越大，模型可以同時考慮的上下文信息就越多。但是，n_head 值過大也會增加模型的計算複雜度。一般來說，n_head 的值可以從 8 到 16 不等。
- d_ff：d_ff 的值越大，前饋網路可以學習的非線性關係就越複雜。但是，d_ff 值過大也會增加模型的計算複雜度。一般來說，d_ff 的值可以從 1024 到 4096 不等。
- num_classes：num_classes 的值表示輸出的類別數。

您可以根據自己的數據集和任務來調整這些參數。例如，如果您的數據集很大，您可以將 d_model 和 d_ff 的值設置得更大。如果您的任務需要考慮大量的上下文信息，您可以將 n_head 的值設置得更大。

目前測試下來參數量:
d_model = 6000 | 1,012,402,690
d_model = 2048 | 163,623,074
d_model = 512 | 22,048,418
