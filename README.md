# سیستم هوشمند پیشنهاد لینک‌سازی داخلی صفحات وب فارسی
## Design and Evaluation of an Internal Linking Recommendation System for Persian Websites Using Semantic Content Analysis

پروژه دانشگاهی مقطع کارشناسی مهندسی کامپیوتر / علوم داده / هوش مصنوعی.

این نرم‌افزار به صورت خودکار سایت‌مپ (Sitemap XML) یک وب‌سایت فارسی را دریافت کرده، صفحات آن را خزش (Crawl) می‌کند، متون اصلی آن را استخراج و نرمال‌سازی پردازش زبان طبیعی (Persian NLP) می‌نماید و با استفاده از دو الگوریتم **TF-IDF (آماری)** و **Sentence-Transformers / Multilingual Sentence-BERT (معنایی)**، شباهت موضوعی بین صفحات را محاسبه کرده و پیشنهادهای لینک‌سازی داخلی بهینه ارائه می‌دهد.

---

## 🌟 ویژگی‌های کلیدی سیستم

1. **پشتیبانی کامل از سایت‌مپ (Sitemap Parser):**
   - دریافت Sitemap XML استاندارد، Sitemap Index و سایت‌مپ‌های تو در تو (Nested)
   - پشتیبانی از فایل‌های فشرده `sitemap.xml.gz`
   - فیلتر کردن هوشمند الگوهای URL (دسته‌بندی‌ها، برچسب‌ها، نویسندگان، تصاویر و PDFها)
   
2. **خزشگر اختصاصی (Web Crawler & Content Extractor):**
   - استخراج محتوای اصلی بدنه مقاله (`<main>` یا `<article>`)
   - حذف خودکار عناصر زاید سایت مانند هدر، فوتر، منوها، تبلیغات و اسکریپت‌ها
   - استخراج متادیتا، عنوان، H1، تعداد کلمات و آدرس کانونی (Canonical URL)

3. **پیش‌پردازش و نرمال‌سازی متون فارسی (Persian NLP Pipeline):**
   - اصلاح کاراکترهای عربی (ي -> ی، ك -> ک)، کاراکترهای نیم‌فاصله (\u200c) و اعداد
   - حذف کلمات ایستای فارسی (Stopwords) و ریشه‌یابی/بن‌واژه‌سازی (Lemmatization) با کتابخانه Hazm

4. **مدل‌های محاسبه شباهت (Similarity Engine):**
   - **TF-IDF + Cosine Similarity:** الگوریتم آماری پایه با قابلیت استخراج کلیدواژه‌های مشترک برای شفافیت (Explainability)
   - **Sentence-Transformers (Multilingual Sentence-BERT):** الگوریتم معنایی با تکنیک تکه‌تکه‌سازی متون طولانی (Chunking) و میانگین‌گیری (Mean Pooling)
   - **Hybrid Score:** الگوریتم ترکیبی وزن‌دار $hybrid = \alpha \cdot semantic + (1 - \alpha) \cdot tfidf$

5. **شناسایی صفحات یتیم و گراف لینک‌ها (Orphan Pages & NetworkX):**
   - کشف صفحات بدون لینک ورودی (Orphan Pages) و صفحات با لینک‌دهی ضعیف
   - مدلسازی گراف جهتداری لینک‌ها و محاسبه PageRank و In-Degree / Out-Degree

6. **ارزیابی دستی و معیارهای دانشگاهی (Academic Evaluation & Precision@K):**
   - ثبت دستی بازخورد مرتبط بودن پیشنهادها (Relevant, Partially Relevant, Not Relevant)
   - محاسبه خودکار معیار Precision@K برای مقایسه علمی روش‌های TF-IDF، معنایی و ترکیبی

7. **داشبورد تعاملی فارسی (RTL Streamlit UI):**
   - رابط کاربری مدرن، راست‌چین و فارسی با نمودارهای Plotly
   - خروجی اکسل چندبرگه (Multi-tab Excel Report) و CSV با فرمت UTF-8-SIG

---

## 🏗️ معماری و ساختار پروژه

```text
Link_sug/
├── app.py                      # برنامه اصلی و رابط کاربری Streamlit
├── requirements.txt            # لیست پیش‌نیازهای پروژه
├── README.md                   # مستندات کامل پروژه
├── .gitignore                  # تنظیمات گیت
│
├── config/                     # تنظیمات پایه و ثابت‌ها
│   ├── __init__.py
│   └── settings.py
│
├── src/                        # سورس‌کد اصلی پروژه
│   ├── __init__.py
│   ├── models.py               # مدل‌های داده (Dataclasses)
│   │
│   ├── crawler/                # ماژول خزش و سایت‌مپ
│   │   ├── sitemap_parser.py
│   │   ├── page_fetcher.py
│   │   └── content_extractor.py
│   │
│   ├── nlp/                    # ماژول پردازش زبان طبیعی فارسی
│   │   ├── persian_normalizer.py
│   │   ├── text_preprocessor.py
│   │   ├── tfidf_model.py
│   │   └── semantic_model.py
│   │
│   ├── analysis/               # ماژول تحلیل شباهت و پیشنهاد
│   │   ├── similarity.py
│   │   ├── link_extractor.py
│   │   ├── recommendation_engine.py
│   │   ├── orphan_detector.py
│   │   └── graph_analyzer.py
│   │
│   ├── evaluation/             # ماژول ارزیابی دانشگاهی
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── export/                 # ماژول گزارش‌گیری و خروجی
│   │   ├── csv_exporter.py
│   │   └── excel_exporter.py
│   │
│   └── utils/                  # ابزارها و توابع کمکی
│       ├── validators.py
│       ├── url_utils.py
│       └── helpers.py
│
├── data/                       # داده‌ها و کش مدل‌ها
│   ├── cache/
│   ├── exports/
│   └── samples/
│
└── tests/                      # تست‌های واحد (Unit Tests)
    ├── test_persian_normalizer.py
    ├── test_url_utils.py
    └── test_recommendation_engine.py
```

---

## 🚀 راهنمای نصب و اجرا

### پیش‌نیازها
- سیستم‌عامل: Windows / Linux / macOS
- پایتون: نسخه 3.11 یا بالاتر (`Python 3.11+`)

### ۱. ساخت محیط مجازی (Virtual Environment)

**در ویندوز (Windows PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**در لینوکس / مک (Linux / macOS):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### ۲. نصب پیش‌نیازها

```bash
pip install -r requirements.txt
```

### ۳. اجرای برنامه Streamlit

```bash
streamlit run app.py
```

پس از اجرای این دستور، مرورگر شما به صورت خودکار آدرس `http://localhost:8501` را باز خواهد کرد.

---

## 🧪 اجرای تست‌های واحد (Unit Tests)

جهت بررسی صحت عملکرد توابع و الگوریتم‌های پروژه دستور زیر را اجرا کنید:

```bash
python -m pytest
```

---

## ⚠️ محدودیت‌های پژوهشی پروژه (Project Limitations)

1. **عدم تغییر مستقیم سایت:** این سیستم صرفاً به عنوان یک ابزار پیشنهادی و تحلیلی عمل می‌کند و آدرس‌ها یا لینک‌های سایت اصلی را به صورت خودکار ویرایش نمی‌کند.
2. **تضمین رتبه گوگل:** شباهت معنایی به معنای تایید ۱۰۰٪ موتورهای جستجو نیست؛ لینک‌سازی باید بر اساس استراتژی تجربه کاربری و سئو انسانی تایید شود.
3. **محدودیت خزش:** پوشش تحلیل وابسته به تعداد صفحات قابل دسترس در سایت‌مپ و پاسخگویی سرور مقصد است.
4. **ارزیابی کیفی:** سنجش دقیق عملکرد الگوریتم‌ها نیازمند ثبت بازخورد در بخش «ارزیابی دستی» توسط کارشناس محتوا می‌باشد.
