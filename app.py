"""
Streamlit Web Application: Persian Internal Link Recommendation System
Simplified, clean, and intuitive user interface with live step-by-step progress tracking.
"""

import os
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Import core modules
from config.settings import (
    DEFAULT_CRAWL_DELAY,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_MAX_URLS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EXPORTS_DIR,
)
from src.analysis.link_extractor import LinkExtractor
from src.analysis.orphan_detector import OrphanDetector
from src.analysis.recommendation_engine import RecommendationEngine
from src.analysis.similarity import compute_hybrid_similarity_matrix
from src.crawler.content_extractor import ContentExtractor
from src.crawler.page_fetcher import PageFetcher
from src.crawler.sitemap_parser import SitemapParser
from src.evaluation.evaluator import EvaluationStore
from src.evaluation.metrics import compute_evaluation_summary, compute_precision_at_k
from src.export.csv_exporter import CSVExporter
from src.export.excel_exporter import ExcelExporter
from src.models import ExperimentConfig, InternalLinkData, PageData, RecommendationData
from src.nlp.persian_normalizer import PersianNormalizer
from src.nlp.semantic_model import SemanticSimilarityModel
from src.nlp.text_preprocessor import TextPreprocessor
from src.nlp.tfidf_model import TFIDFSimilarityModel
from src.utils.helpers import format_persian_number, setup_logger
from src.utils.validators import is_valid_url

logger = setup_logger("streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="پیشنهاددهنده هوشمند لینک داخلی",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom RTL Persian Styling
RTL_CSS = """
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    html, body, [class*="css"], .stMarkdown, .stText, .stButton, div {
        font-family: 'Vazirmatn', Tahoma, sans-serif !important;
        direction: rtl;
        text-align: right;
    }
    
    .stApp {
        direction: rtl;
    }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: #f8fafc;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .metric-card p {
        font-size: 1.6rem;
        font-weight: bold;
        color: #38bdf8;
        margin: 0;
    }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)


# Session State Initialization
if "crawled_pages" not in st.session_state:
    st.session_state.crawled_pages = []
if "existing_links" not in st.session_state:
    st.session_state.existing_links = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "experiment_config" not in st.session_state:
    st.session_state.experiment_config = None

eval_store = EvaluationStore()


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ تنظیمات ورودی")
    
    sitemap_input = st.text_input(
        "آدرس سایت‌مپ (Sitemap URL)",
        value="https://example.com/sitemap.xml",
        help="آدرس اصلی sitemap.xml یا sitemap_index.xml سایت خود را وارد کنید.",
    )
    
    max_pages = st.slider(
        "تعداد صفحات برای بررسی",
        min_value=5,
        max_value=150,
        value=30,
        step=5,
    )

    selected_method = st.selectbox(
        "روش هوشمند الگوریتم",
        options=["hybrid", "semantic", "tfidf"],
        format_func=lambda x: {
            "hybrid": "ترکیبی هوشمند (پیشنهادی)",
            "semantic": "معنایی (Sentence-BERT)",
            "tfidf": "آماری (TF-IDF)",
        }[x],
        index=0,
    )

    top_k = st.slider("تعداد پیشنهاد برای هر صفحه", min_value=1, max_value=10, value=5)
    
    similarity_thresh = st.slider(
        "حداقل آستانه شباهت",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
    )

    exclude_existing = st.checkbox("حذف لینک‌های موجود از پیشنهادها", value=True)

    start_btn = st.button("🚀 شروع تحلیل سایت", type="primary", use_container_width=True)


# --- MAIN PIPELINE ACTION WITH LIVE PROGRESS ---
if start_btn:
    if not sitemap_input or not sitemap_input.startswith("http"):
        st.error("لطفاً یک آدرس سایت‌مپ معتبر (همراه با http/https) وارد کنید.")
    else:
        is_valid, err_msg = is_valid_url(sitemap_input)
        if not is_valid:
            st.error(err_msg)
        else:
            status_box = st.status("⏳ در حال اجرای مراحل تحلیل سایت...", expanded=True)
            
            with status_box:
                # 1. Parse Sitemap
                st.write("🔍 **مرحله ۱ از ۵:** در حال استخراج آدرس‌ها از سایت‌مپ...")
                parser = SitemapParser(
                    sitemap_url=sitemap_input,
                    timeout=DEFAULT_REQUEST_TIMEOUT,
                    max_urls=max_pages,
                    exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
                )
                discovered, sitemap_errors = parser.parse()

                if not discovered:
                    st.error("❌ هیچ آدرسی از سایت‌مپ استخراج نشد.")
                    for err in sitemap_errors:
                        st.warning(err)
                    st.stop()

                st.success(f"✅ تعداد {len(discovered)} صفحه از سایت‌مپ استخراج شد.")

                # 2. Crawl & Content Extraction
                st.write("🌐 **مرحله ۲ از ۵:** در حال خزش صفحات و خواندن متن داخل مقاله...")
                fetcher = PageFetcher(
                    base_domain_url=discovered[0]["url"],
                    timeout=DEFAULT_REQUEST_TIMEOUT,
                    crawl_delay=DEFAULT_CRAWL_DELAY,
                )

                crawled_pages: List[Dict[str, Any]] = []
                all_existing_links: List[InternalLinkData] = []
                
                progress_bar = st.progress(0)
                live_text = st.empty()

                for idx, item in enumerate(discovered):
                    url = item["url"]
                    live_text.info(f"در حال دریافت صفحه [{idx + 1}/{len(discovered)}]: {url}")
                    
                    html_content, status_code, fetch_err = fetcher.fetch(url)

                    if html_content and status_code == 200:
                        content_info = ContentExtractor.extract(html_content, page_url=url)
                        
                        page_dict = {
                            "page_id": idx + 1,
                            "url": url,
                            "title": content_info["title"] or url,
                            "meta_description": content_info["meta_description"],
                            "h1": content_info["h1"],
                            "raw_text": content_info["raw_text"],
                            "clean_text": content_info["clean_text"],
                            "word_count": content_info["word_count"],
                            "status_code": status_code,
                            "crawl_status": "success",
                            "canonical_url": content_info["canonical_url"],
                        }
                        crawled_pages.append(page_dict)

                        # Extract existing links ONLY inside main content body
                        links = LinkExtractor.extract_internal_links(
                            html_content=html_content,
                            source_url=url,
                            base_domain_url=discovered[0]["url"],
                        )
                        all_existing_links.extend(links)
                    else:
                        crawled_pages.append({
                            "page_id": idx + 1,
                            "url": url,
                            "title": url,
                            "meta_description": "",
                            "h1": "",
                            "raw_text": "",
                            "clean_text": "",
                            "word_count": 0,
                            "status_code": status_code,
                            "crawl_status": "failed",
                            "crawl_error": fetch_err or "خطا در خزش",
                        })

                    progress_bar.progress((idx + 1) / len(discovered))

                live_text.empty()
                valid_pages = [p for p in crawled_pages if p["crawl_status"] == "success" and p["word_count"] >= 1]
                st.session_state.crawled_pages = valid_pages
                st.session_state.existing_links = all_existing_links

                if len(valid_pages) < 2:
                    failed_count = sum(1 for p in crawled_pages if p["crawl_status"] != "success")
                    st.error(f"❌ تعداد صفحات موفق با محتوای کافی کمتر از ۲ صفحه است (تعداد کل صفحات: {len(crawled_pages)}، ناموفق: {failed_count}).")
                    st.write("🔍 **جزئیات خطاهای رخ داده در خزش:**")
                    for p in crawled_pages:
                        if p["crawl_status"] != "success":
                            st.warning(f"صفحه {p['url']} - کد وضعیت: {p.get('status_code', 0)} - خطا: {p.get('crawl_error', 'محتوای متنی کافی یافت نشد')}")
                    st.stop()

                st.success(f"✅ تعداد {len(valid_pages)} صفحه با موفقیت تحلیل گردید. (تعداد {len(all_existing_links)} لینک داخلی واقعی از متن استخراج شد)")

                # 3. TF-IDF Similarity
                st.write("📝 **مرحله ۳ از ۵:** نرمال‌سازی متون فارسی و محاسبه مدل آماری TF-IDF...")
                tfidf_model = TFIDFSimilarityModel()
                tfidf_matrix = tfidf_model.fit_transform(valid_pages)

                # 4. Semantic Similarity
                st.write("🤖 **مرحله ۴ از ۵:** محاسبه تعبیه‌های معنایی متون (Sentence Transformers)...")
                semantic_model = SemanticSimilarityModel(model_name=DEFAULT_SEMANTIC_MODEL)
                semantic_matrix, sem_err = semantic_model.compute_similarity_matrix(valid_pages)
                if sem_err:
                    semantic_matrix = tfidf_matrix

                # 5. Hybrid Matrix & Recommendations
                st.write("💡 **مرحله ۵ از ۵:** ساخت و رتبه‌بندی پیشنهادهای لینک‌سازی داخلی...")
                hybrid_matrix = compute_hybrid_similarity_matrix(
                    tfidf_matrix=tfidf_matrix,
                    semantic_matrix=semantic_matrix,
                    alpha=DEFAULT_HYBRID_ALPHA,
                )

                existing_set = {(l.source_url, l.target_url) for l in all_existing_links}
                
                recommendations = RecommendationEngine.generate_recommendations(
                    pages=valid_pages,
                    tfidf_matrix=tfidf_matrix,
                    semantic_matrix=semantic_matrix,
                    hybrid_matrix=hybrid_matrix,
                    existing_links_set=existing_set,
                    method=selected_method,
                    top_k=top_k,
                    similarity_threshold=similarity_thresh,
                    exclude_existing_links=exclude_existing,
                    tfidf_model=tfidf_model,
                )
                st.session_state.recommendations = recommendations

                config_rec = ExperimentConfig(
                    sitemap_url=sitemap_input,
                    max_urls=max_pages,
                    selected_method=selected_method,
                    embedding_model=DEFAULT_SEMANTIC_MODEL,
                    hybrid_alpha=DEFAULT_HYBRID_ALPHA,
                    similarity_threshold=similarity_thresh,
                    top_k=top_k,
                    discovered_urls_count=len(discovered),
                    crawled_pages_count=len(valid_pages),
                )
                st.session_state.experiment_config = config_rec

                status_box.update(label="🎉 تحلیل سایت با موفقیت به پایان رسید!", state="complete")


# --- MAIN HEADER & SIMPLIFIED TABS ---
st.title("🔗 سیستم هوشمند پیشنهاد لینک‌سازی داخلی")

pages = st.session_state.crawled_pages
recommendations = st.session_state.recommendations
existing_links = st.session_state.existing_links

tab_recs, tab_page, tab_orphan, tab_dash, tab_export = st.tabs([
    "💡 جدول پیشنهادها",
    "🔍 بررسی صفحه به صفحه",
    "🏝️ صفحات بدون لینک",
    "📊 آمار و خلاصه",
    "📥 دریافت گزارش",
])


# --- TAB 1: RECOMMENDATIONS TABLE ---
with tab_recs:
    st.subheader("💡 جدول بهترین پیشنهادهای لینک‌سازی داخلی")
    if not recommendations:
        st.info("برای شروع تحلیل سایت، آدرس سایت‌مپ را در نوار کناری وارد کرده و دکمه «شروع تحلیل سایت» را بزنید.")
    else:
        rec_list = [r.to_dict() for r in recommendations]
        df_rec = pd.DataFrame(rec_list)

        search_query = st.text_input("🔍 جستجو در عناوین صفحات:", value="")
        if search_query:
            df_rec = df_rec[
                df_rec["source_title"].str.contains(search_query, case=False, na=False)
                | df_rec["target_title"].str.contains(search_query, case=False, na=False)
            ]

        st.dataframe(
            df_rec[[
                "rank", "source_title", "target_title", "similarity_score",
                "similarity_category", "existing_link", "common_terms"
            ]].rename(columns={
                "rank": "رتبه",
                "source_title": "صفحه مبدا",
                "target_title": "صفحه پیشنهادی (هدف)",
                "similarity_score": "امتیاز شباهت",
                "similarity_category": "سطح شباهت",
                "existing_link": "لینک فعلی دارد؟",
                "common_terms": "کلمات مشترک",
            }),
            use_container_width=True,
        )


# --- TAB 2: PAGE BY PAGE ---
with tab_page:
    st.subheader("🔍 تحلیل و پیشنهاد اختصاصی برای یک صفحه")
    if not pages:
        st.info("اطلاعات صفحه‌ای موجود نیست.")
    else:
        url_options = {p["url"]: f"{p['title']}" for p in pages}
        selected_url = st.selectbox("انتخاب صفحه مورد نظر:", options=list(url_options.keys()), format_func=lambda x: url_options[x])

        page_info = next((p for p in pages if p["url"] == selected_url), None)
        if page_info:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**عنوان:** {page_info['title']}")
                st.write(f"**آدرس:** [{page_info['url']}]({page_info['url']})")
            with c2:
                st.write(f"**تعداد کلمات متن:** {format_persian_number(page_info['word_count'])}")

            st.divider()
            st.markdown(f"#### 💡 بهترین پیشنهادهای لینک‌دهی از این صفحه:")

            page_recs = [r for r in recommendations if r.source_url == selected_url]
            if not page_recs:
                st.info("پیشنهادی برای این صفحه یافت نشد.")
            else:
                for r in page_recs:
                    st.success(f"**رتبه {r.rank}:** {r.target_title}\n\n**امتیاز شباهت:** {r.similarity_score:.3f} | **آدرس هدف:** {r.target_url}")


# --- TAB 3: ORPHAN PAGES ---
with tab_orphan:
    st.subheader("🏝️ صفحات بدون لینک ورودی (Orphan Pages)")
    if not pages:
        st.info("داده‌ای برای تحلیل صفحات یتیم وجود ندارد.")
    else:
        orphan_analysis = OrphanDetector.analyze_link_counts(pages, existing_links)
        
        df_orphans = pd.DataFrame(orphan_analysis["orphans"])
        if df_orphans.empty:
            st.success("🎉 هیچ صفحه یتیمی در متن مقالات خزش‌شده پیدا نشد!")
        else:
            st.warning(f"تعداد {len(df_orphans)} صفحه بدون هیچ لینک ورودی از داخل متن مقالات شناسایی شد:")
            st.dataframe(
                df_orphans[["title", "url", "word_count"]].rename(columns={
                    "title": "عنوان صفحه",
                    "url": "آدرس URL",
                    "word_count": "تعداد کلمات",
                }),
                use_container_width=True,
            )


# --- TAB 4: SUMMARY & STATS ---
with tab_dash:
    st.subheader("📊 خلاصه وضعیت و آمار عمومی")
    if not pages:
        st.info("هنوز تحلیلی انجام نشده است.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        orphan_analysis = OrphanDetector.analyze_link_counts(pages, existing_links)

        with col1:
            st.markdown(f'<div class="metric-card"><h3>صفحات تحلیل‌شده</h3><p>{format_persian_number(len(pages))}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>لینک‌های فعلی (داخل متن)</h3><p>{format_persian_number(len(existing_links))}</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>پیشنهادهای تولیدشده</h3><p>{format_persian_number(len(recommendations))}</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>صفحات یتیم</h3><p>{format_persian_number(orphan_analysis["total_orphans"])}</p></div>', unsafe_allow_html=True)

        st.divider()
        if recommendations:
            scores = [r.similarity_score for r in recommendations]
            fig = px.histogram(x=scores, labels={"x": "امتیاز شباهت", "y": "تعداد پیشنهاد"}, title="توزیع امتیاز شباهت پیشنهادها")
            st.plotly_chart(fig, use_container_width=True)


# --- TAB 5: EXPORT ---
with tab_export:
    st.subheader("📥 دریافت خروجی نهایی گزارش")
    if not pages:
        st.info("داده‌ای برای خروجی گرفتن وجود ندارد.")
    else:
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.markdown("### 📊 فایل کامل Excel (.xlsx)")
            excel_path = ExcelExporter.export_all(
                summary_data=st.session_state.experiment_config.to_dict() if st.session_state.experiment_config else {},
                pages_data=pages,
                recommendations_data=[r.to_dict() for r in recommendations],
                orphan_data=OrphanDetector.analyze_link_counts(pages, existing_links)["orphans"],
                existing_links_data=[l.to_dict() for l in existing_links],
                evaluation_data=[e.to_dict() for e in eval_store.get_all()],
            )
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="⬇️ دانلود فایل اکسل (Excel)",
                    data=f.read(),
                    file_name=excel_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        with col_ex2:
            st.markdown("### 📄 فایل CSV")
            if recommendations:
                csv_path = CSVExporter.export_to_csv([r.to_dict() for r in recommendations], "recommendations.csv")
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="⬇️ دانلود فایل CSV",
                        data=f.read(),
                        file_name="persian_internal_link_recommendations.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
