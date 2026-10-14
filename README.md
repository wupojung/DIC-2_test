# 50 Startups — Multiple Linear Regression (多元線性迴歸分析)

本專案依據 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 資料探勘方法論的六大階段，使用 **Python** 與 **Scikit-learn** 建置 **Multiple Linear Regression (多元線性迴歸)** 模型，預測新創公司的 **Profit (淨利潤)**。

---

## 專案目錄結構

```text
DIC-2/
├── .venv/                      # Python 虛擬環境
├── data/
│   └── 50_Startups.csv         # 原始資料集 (50 筆記錄)
├── outputs/
│   ├── actual_vs_predicted.png # 實際值 vs. 預測值對比圖
│   ├── correlation_heatmap.png # 特徵相關係數熱力圖 (Heatmap)
│   ├── eda_distributions.png   # 數值特徵與目標變數分布直方圖
│   ├── eda_scatter_relationships.png # 各支出特徵對利潤的散佈圖
│   ├── evaluation_summary.txt  # 評估指標與迴歸方程式純文字摘要
│   ├── feature_selection_comparison.png # 五大特徵選擇演算法對比圖
│   ├── feature_selection_performance.png # 特徵子集模型效能對比圖
│   ├── feature_selection_results.json   # 特徵子集效能 JSON 記錄
│   ├── metrics.json            # 完整模型指標與參數 JSON 檔
│   ├── model_pipeline.joblib   # 序列化封裝之完整預測 Pipeline
│   └── residual_plot.png       # 殘差圖 (Residual Plot) 與殘差分布圖
├── feature_selection.py        # 5 種流行特徵選擇演算法分析模組
├── main.py                     # CRISP-DM 六階段完整執行主程式
├── predict.py                  # 模型載入與新資料推論預測腳本
├── requirements.txt            # Python 依賴套件清單
├── README.md                   # 專案快速說明文件 (繁體中文)
└── REPORT.md                   # 完整圖文並茂專業分析報告 (包含所有圖片顯示)
```

---

## 環境建置與安裝步驟

建議使用 Python 3.12 或相容版本 (如 Python 3.13) 建立虛擬環境：

### 1. 建立並啟用虛擬環境 (`.venv`)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安裝必要套件

```bash
pip install -r requirements.txt
```

`requirements.txt` 內容包含：
- `pandas`：資料讀取、清理與結構化分析
- `numpy`：數值計算與向量運算
- `matplotlib`：資料視覺化與診斷圖表輸出
- `scikit-learn`：特徵轉換 Pipeline、多元線性迴歸模型與交叉驗證評估
- `joblib`：模型 Pipeline 序列化與儲存

---

## CRISP-DM 六階段執行過程與成果

### Phase 1: Business Understanding (商業理解)

- **商業目標**：協助創投機構 (Venture Capital) 與新創管理團隊評估研發、行政與行銷三項核心支出及地理位置對淨利潤的影響程度，以作為投資決策與預算配置的最佳化依據。
- **預測目標 (Target Variable)**：
  - `Profit` (float64)：新創公司的年度淨利潤 (單位：USD)。
- **輸入特徵 (Predictor Features)**：
  - `R&D Spend` (float64)：研發支出費用 (USD)。
  - `Administration` (float64)：行政與營運開銷 (USD)。
  - `Marketing Spend` (float64)：市場行銷與廣告支出 (USD)。
  - `State` (object / category)：公司註冊與營運州別，包含 `New York`、`California`、`Florida`。

---

### Phase 2: Data Understanding (資料理解)

1. **基本資料結構**：
   - 資料筆數：共 50 筆樣本，5 個欄位。
   - 欄位型態：4 個數值連續型欄位 (`float64`) 與 1 個類別欄位 (`object` / `str`)。
   - 缺失值檢查：所有欄位缺失值均為 0 (無遺漏值)。
   - 州別分佈：New York (17 筆)、California (17 筆)、Florida (16 筆)，三地樣本分佈均衡。

2. **統計摘要 (Descriptive Statistics)**：
   - `R&D Spend`：平均值 $73,721.62，標準差 $45,902.26，最小值 $0.00，最大值 $165,349.20。
   - `Administration`：平均值 $121,344.64，標準差 $28,017.80，最小值 $51,283.14，最大值 $182,645.56。
   - `Marketing Spend`：平均值 $211,025.10，標準差 $122,290.31，最小值 $0.00，最大值 $471,784.10。
   - `Profit`：平均值 $112,012.64，標準差 $40,306.18，最小值 $14,681.40，最大值 $192,261.83。

3. **皮爾森相關係數分析 (Pearson Correlation Matrix)**：
   | 欄位名稱 | R&D Spend | Administration | Marketing Spend | Profit |
   | :--- | :---: | :---: | :---: | :---: |
   | **R&D Spend** | 1.0000 | 0.2420 | 0.7242 | **0.9729** |
   | **Administration** | 0.2420 | 1.0000 | -0.0322 | 0.2007 |
   | **Marketing Spend** | 0.7242 | -0.0322 | 1.0000 | **0.7478** |
   | **Profit** | **0.9729** | 0.2007 | **0.7478** | 1.0000 |

   - **關鍵發現**：`R&D Spend` 與 `Profit` 高度正相關 ($r = 0.9729$)，為最具預測力的特徵；`Marketing Spend` 次之 ($r = 0.7478$)；`Administration` 與利潤相關性低 ($r = 0.2007$)。

4. **輸出圖表**：
   - `outputs/correlation_heatmap.png`：數值特徵相關性熱力圖。
   - `outputs/eda_distributions.png`：特徵與目標變數之直方圖分布。
   - `outputs/eda_scatter_relationships.png`：各支出特徵與利潤依州別標色的散佈圖。

---

### Phase 3: Data Preparation (資料準備)

1. **缺失值處理**：程式自動驗證缺失值數量，本資料集共 0 筆缺失值。
2. **類別型特徵編碼 (One-Hot Encoding)**：
   - 針對 `State` 欄位使用 `OneHotEncoder(drop='first', sparse_output=False)`。
   - 設定 `drop='first'` 自動剔除基準組 (Baseline category: `California`)，保留 `State_Florida` 與 `State_New York` 兩欄虛擬變數，有效避免多元線性迴歸中的 **Dummy Variable Trap (虛擬變數陷阱 / 多重共線性)**。
3. **資料切分 (Train/Test Split)**：
   - 採用 **80% Training Set (40 筆)** 與 **20% Testing Set (10 筆)**。
   - 設定 `random_state=42` 確保實驗具備完全可重複性 (Reproducibility)。
4. **防止資料洩漏 (Data Leakage Prevention)**：
   - 使用 Scikit-learn 的 `ColumnTransformer` 與 `Pipeline` 將資料預處理與迴歸演算法封裝成單一物件。
   - 預處理器的擬合 (Fit) 嚴格限定於 Training Set (或 Cross-Validation 各折的訓練子集)，測試集僅透過已擬合之轉換器進行推論，徹底防止資料洩漏。

---

### Phase 4: Modeling (模型建置)

1. **演算法選擇**：使用 `sklearn.linear_model.LinearRegression` 建置 Multiple Linear Regression 模型。
2. **5-Fold Cross-Validation (5 折交叉驗證)**：
   - 在 40 筆訓練集樣本上採用 `KFold(n_splits=5, shuffle=True, random_state=42)`。
   - 各折驗證表現結果：
     - **Fold 1**：$R^2 = 0.9539$，$\text{MAE} = \$7,946.61$，$\text{RMSE} = \$9,794.97$
     - **Fold 2**：$R^2 = 0.9543$，$\text{MAE} = \$7,175.21$，$\text{RMSE} = \$8,777.27$
     - **Fold 3**：$R^2 = 0.8933$，$\text{MAE} = \$10,346.96$，$\text{RMSE} = \$14,900.98$
     - **Fold 4**：$R^2 = 0.9713$，$\text{MAE} = \$4,914.89$，$\text{RMSE} = \$6,091.63$
     - **Fold 5**：$R^2 = 0.8972$，$\text{MAE} = \$6,745.75$，$\text{RMSE} = \$8,188.53$
   - **5-Fold CV 平均指標**：
     - 平均決定係數 $\text{Mean } R^2 = \mathbf{0.9340}$ (標準差 $\text{std} = 0.0323$)
     - 平均絕對誤差 $\text{Mean MAE} = \mathbf{\$7,425.88}$ (標準差 $\text{std} = \$1,768.30$)
     - 均方根誤差 $\text{Mean RMSE} = \mathbf{\$9,550.68}$ (標準差 $\text{std} = \$2,936.12$)
3. **全訓練集擬合**：
   - 交叉驗證確認模型穩定後，在完整 80% 訓練集上擬合最終模型 Pipeline。

---

### Phase 5: Evaluation (模型評估)

在未參與訓練的 20% 獨立測試集 (Holdout Test Set, $N = 10$) 上評估模型績效：

#### 1. 測試集量化評估指標 (Test Set Performance)

| 評估指標 | 縮寫 | 數值 | 說明 |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error** | **MAE** | **$6,961.48** | 平均預測誤差金額約 6,961 美元 |
| **Mean Squared Error** | **MSE** | **$82,010,363.04** | 誤差平方均值 |
| **Root Mean Squared Error** | **RMSE** | **$9,055.96** | 懲罰較大誤差後的平均誤差金額約 9,056 美元 |
| **Coefficient of Determination** | **$R^2$** | **0.8987** | 模型可解釋測試集近 **90%** 的利潤變異 |

*(對比訓練集：Train $R^2 = 0.9537$、Train RMSE = $8,927.49$，表現均衡穩定，無明顯過擬合情況)*

#### 2. 迴歸方程式與係數詮釋

```text
Profit = 54,028.04
       + 0.8056 * (R&D Spend)
       - 0.0688 * (Administration)
       + 0.0299 * (Marketing Spend)
       + 938.7930 * (State_Florida)
       + 6.9878 * (State_New York)
```

- **截距項 (Intercept $\beta_0 = \$54,028.04$)**：基準組 (位於 California 且各項支出為 0) 的基礎預期利潤。
- **R&D Spend ($\beta_1 = 0.8056$)**：其他條件不變下，每增加 $1 美元的研發投入，預期利潤提升約 **$0.81 美元**，效益最為顯著。
- **Administration ($\beta_2 = -0.0688$)**：每增加 $1 美元的行政開銷，預期利潤略微下降約 **$0.07 美元**，反映過高行政成本對獲利產生微幅負面壓力。
- **Marketing Spend ($\beta_3 = 0.0299$)**：每增加 $1 美元的行銷投入，預期利潤增加約 **$0.03 美元**。
- **地理州別係數 (基準為 California)**：
  - `State_Florida` ($\beta_4 = 938.79$)：相較於 California，同等支出條件下 Florida 的新創平均利潤高出約 $938.79 美元。
  - `State_New York` ($\beta_5 = 6.99$)：相較於 California，同等支出條件下 New York 僅高出約 $6.99 美元，兩者差異極小。

#### 3. 視覺化診斷圖表

1. **實際值 vs. 預測值對比圖 (`outputs/actual_vs_predicted.png`)**：
   - 散佈點均勻緊密地分佈於 45 度理想線 ($y = x$) 兩側，證明高、中、低利潤區間皆具備良好的預測擬合度。
2. **殘差診斷圖 (`outputs/residual_plot.png`)**：
   - 殘差對預測值的散佈圖隨機圍繞 0 水平線分佈，未呈現漏斗狀或非線性彎曲，符合線性迴歸之**同質變異性 (Homoscedasticity)** 假定。
   - 殘差直方圖呈現大致對稱的鐘形常態分佈，符合**誤差常態性 (Normality of Residuals)** 假設。

---

### Phase 6: Deployment (模型部署)

1. **模型導出保存**：
   - 將包含特徵預處理 (`ColumnTransformer`) 與迴歸估計器 (`LinearRegression`) 的完整 Pipeline 序列化儲存至 `outputs/model_pipeline.joblib`。
2. **推論腳本 (`predict.py`)**：
   - 支援接收外部字典或 `pandas.DataFrame` 格式的新創公司資料。
   - 支援命令列參數 (CLI) 快速預測單一公司利潤。

---

## 執行方式

### 1. 執行完整 CRISP-DM 流程 (`main.py`)

執行主程式將依序跑完六個階段，並自動將圖表與評估結果輸出至 `outputs/` 目錄：

```bash
python main.py
```

### 2. 執行模型預測 (`predict.py`)

#### (A) 示範模式 (Demo Mode)：
執行預設示範案例，展示不同州別與不同規模新創的預測結果：
```bash
python predict.py
```

#### (B) 命令列參數模式 (CLI Mode)：
可透過命令列指定研發、行政、行銷支出與營運州別，即時計算預測淨利潤：
```bash
python predict.py --rd 130000 --admin 120000 --marketing 320000 --state "New York"
```

**輸出範例：**
```text
Single Startup Profit Prediction:
  R&D Spend:        $130,000.00
  Administration:   $120,000.00
  Marketing Spend:  $320,000.00
  State:            New York
  --> Predicted Profit: $160,066.13
```

### 3. 獨立執行 5 大特徵選擇演算法分析 (`feature_selection.py`)

如需單獨執行特徵選擇演算法分析與繪製對比圖表：
```bash
python feature_selection.py
```

### 4. 查看完整圖文並茂分析報告 (`REPORT.md`)
本專案已生成排版精美、內嵌所有視覺化圖表的完整分析報告：
- 本地 Markdown 預覽：請直接開啟專案根目錄下的 [REPORT.md](file:///d:/AI/DIC-2/REPORT.md)。

---

## 驗證與結果總結

1. **BUILD**：成功建立乾淨的專案目錄結構與相依套件清單 (`requirements.txt`)。
2. **RUN**：完整執行 `main.py`，無錯誤完成所有運算並產出圖表與模型。
3. **TEST**：透過 5-Fold Cross-Validation (Mean $R^2 = 0.9340$) 與獨立測試集驗證 (Test $R^2 = 0.8987$)，模型穩定且具備良好泛化能力。
4. **VERIFY**：使用 `predict.py` 成功載入 `outputs/model_pipeline.joblib`，完成批次推論與 CLI 預測驗證。
5. **PASS**：全階段通過驗證，各項產出檔案與數值指標完整且真實對齊。
