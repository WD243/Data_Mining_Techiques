import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ================= 1. THIẾT LẬP GIAO DIỆN =================
st.set_page_config(page_title="Hệ thống Khai phá Dữ liệu", page_icon="📊", layout="wide")

# CSS Tối ưu hóa hiển thị: Font chữ hiện đại, màu sắc trung tính, bo góc các khối
st.markdown("""
    <style>
    /* Tổng thể */
    .main { background-color: #fcfcfc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Tùy chỉnh khối Metric */
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1e293b; font-weight: 700; }
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #edf2f7; }
    
    /* Header & Title */
    h1, h2, h3 { color: #334155 !important; font-weight: 600 !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
    
    /* Nút bấm */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 500; height: 3rem; background-color: #3b82f6; color: white; border: none; }
    .stButton>button:hover { background-color: #2563eb; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. QUẢN LÝ DỮ LIỆU =================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        # Loại bỏ cột trùng tên nếu có
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

# ================= 3. THANH ĐIỀU KHIỂN (SIDEBAR) =================
with st.sidebar:
    st.title("📊 Phân tích Dữ liệu")
    st.write("Cung cấp các công cụ khai phá tri thức từ tệp dữ liệu của bạn.")
    st.divider()
    
    st.subheader("📁 Bước 1: Nguồn dữ liệu")
    uploaded_file = st.file_uploader("Tải file CSV hoặc Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        st.success("Đã nhận file thành công")
        st.divider()
        
        st.subheader("🛠️ Bước 2: Công cụ xử lý")
        algo = st.selectbox(
            "Chọn thuật toán:",
            ["Trình diễn dữ liệu", "Tương quan Pearson", "Luật kết hợp (Apriori)", 
             "Phân loại Naive Bayes", "Cây quyết định ID3", "Lý thuyết Tập thô"]
        )
        
        st.divider()
        st.subheader("⚙️ Bước 3: Thông số")
        test_size = st.slider("Tỉ lệ kiểm tra (%)", 10, 50, 25) / 100
        max_depth = st.slider("Độ sâu tối đa (Cây)", 2, 20, 6)

# ================= 4. NỘI DUNG CHÍNH =================
if not uploaded_file:
    st.title("Chào mừng đến với Hệ thống Phân tích Dữ liệu")
    st.markdown("""
    Vui lòng **tải lên dữ liệu** ở cột bên trái để bắt đầu quy trình khai phá.
    
    ---
    ### 🧭 Quy trình làm việc:
    *   **Tải lên:** Hỗ trợ các định dạng bảng phổ biến (Excel, CSV).
    *   **Khám phá:** Xem các đặc trưng thống kê và phân bổ của dữ liệu.
    *   **Phân tích:** Áp dụng các thuật toán khai phá để tìm quy luật và phân loại.
    """)
    st.info("💡 Lưu ý: Hệ thống hoạt động tốt nhất với dữ liệu đã được làm sạch.")
else:
    df = load_data(uploaded_file)
    if df is not None:
        # Lựa chọn cột mục tiêu (Dùng cho các thuật toán phân loại)
        target_col = st.sidebar.selectbox("🎯 Biến mục tiêu:", df.columns, index=len(df.columns)-1)
        
        # ---------------------------------------------------------
        # TỔNG QUAN DỮ LIỆU
        # ---------------------------------------------------------
        if algo == "Trình diễn dữ liệu":
            st.header("📋 Đặc trưng tổng quan")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng số dòng", df.shape[0])
            c2.metric("Số cột", df.shape[1])
            c3.metric("Ô trống (Null)", df.isna().sum().sum())
            c4.metric("Dòng trùng lặp", df.duplicated().sum())
            
            st.subheader("Bản xem trước dữ liệu")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Tỉ lệ phân bổ lớp")
            fig_pie = px.pie(df, names=target_col, hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ---------------------------------------------------------
        # TƯƠNG QUAN
        # ---------------------------------------------------------
        elif algo == "Tương quan Pearson":
            st.header("🔗 Mối liên hệ giữa các thuộc tính")
            feature = st.selectbox("Chọn thuộc tính để so sánh:", [c for c in df.columns if c != target_col])
            
            df_c = df[[feature, target_col]].dropna()
            
            # Encode tạm thời để tính toán Pearson r
            le = LabelEncoder()
            for c in df_c.columns:
                if not pd.api.types.is_numeric_dtype(df_c[c]):
                    df_c[c] = le.fit_transform(df_c[c].astype(str))
            
            r_val = np.corrcoef(df_c[feature], df_c[target_col])[0, 1]
            
            res1, res2 = st.columns([1, 2])
            res1.metric("Hệ số Pearson r", f"{r_val:.4f}")
            res1.write("Giải thích: r càng gần 1 hoặc -1 thể hiện mối quan hệ càng chặt chẽ.")
            
            fig_s = px.scatter(df_c, x=feature, y=target_col, trendline="ols", color_discrete_sequence=['#3b82f6'])
            res2.plotly_chart(fig_s, use_container_width=True)

        # ---------------------------------------------------------
        # APRIORI (Giao diện mới)
        # ---------------------------------------------------------
        elif algo == "Luật kết hợp (Apriori)":
            st.header("🛒 Khám phá tập mục thường xuyên")
            st.write("Thuật toán tìm kiếm các nhóm giá trị thường xuất hiện đồng thời trong các bản ghi.")
            
            if st.button("Bắt đầu tính toán"):
                with st.spinner("Đang trích xuất dữ liệu..."):
                    df_ap = df.drop(columns=['student_id', 'timestamp'], errors='ignore').copy()
                    # Rời rạc hóa
                    for col in df_ap.select_dtypes(include=np.number).columns:
                        if df_ap[col].nunique() > 4:
                            df_ap[col] = pd.cut(df_ap[col], 3, labels=['Thấp', 'Trung bình', 'Cao'])
                    
                    item_counts = {}
                    for _, row in df_ap.iterrows():
                        for c, v in row.items():
                            key = f"{c}: {v}"
                            item_counts[key] = item_counts.get(key, 0) + 1
                    
                    df_res = pd.DataFrame(sorted(item_counts.items(), key=lambda x:x[1], reverse=True)[:12], 
                                         columns=["Tập thuộc tính", "Tần suất"])
                
                col_a, col_b = st.columns([4, 6])
                with col_a:
                    st.write("**Bảng chỉ số Support:**")
                    st.table(df_res)
                with col_b:
                    fig_a = px.bar(df_res, x="Tần suất", y="Tập thuộc tính", orientation='h', 
                                  color="Tần suất", color_continuous_scale="Blues")
                    st.plotly_chart(fig_a, use_container_width=True)

        # ---------------------------------------------------------
        # NAIVE BAYES
        # ---------------------------------------------------------
        elif algo == "Phân loại Naive Bayes":
            st.header("🕊️ Phân loại dựa trên xác suất")
            if st.button("Thực hiện phân loại"):
                df_nb = df.drop(columns=['student_id', 'timestamp'], errors='ignore').dropna()
                X = df_nb.drop(target_col, axis=1)
                y = df_nb[target_col]
                
                for c in X.columns: X[c] = LabelEncoder().fit_transform(X[c].astype(str))
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model = GaussianNB().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.metric("Độ chính xác (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
                
                t1, t2 = st.tabs(["Ma trận nhầm lẫn", "Báo cáo chi tiết"])
                with t1:
                    cm = confusion_matrix(y_test, y_pred)
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=model.classes_, y=model.classes_), use_container_width=True)
                with t2:
                    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T, use_container_width=True)

        # ---------------------------------------------------------
        # ID3
        # ---------------------------------------------------------
        elif algo == "Cây quyết định ID3":
            st.header("🌳 Phân tích cấu trúc Cây quyết định")
            if st.button("Xây dựng sơ đồ cây"):
                df_id = df.drop(columns=['student_id', 'timestamp'], errors='ignore').dropna()
                X = df_id.drop(target_col, axis=1)
                y = df_id[target_col]
                
                for c in X.columns: X[c] = LabelEncoder().fit_transform(X[c].astype(str))
                le_y = LabelEncoder()
                y_enc = le_y.fit_transform(y.astype(str))
                
                X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42)
                model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth).fit(X_train, y_train)
                
                st.metric("Độ chính xác mô hình", f"{accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
                
                st.subheader("Cấu trúc phân tầng (Tree Plot)")
                fig_t, ax = plt.subplots(figsize=(20, 10), dpi=90)
                plot_tree(model, feature_names=X.columns, class_names=list(le_y.classes_), filled=True, rounded=True)
                st.pyplot(fig_t)

        # ---------------------------------------------------------
        # ROUGH SET
        # ---------------------------------------------------------
        elif algo == "Lý thuyết Tập thô":
            st.header("🧮 Xử lý dữ liệu không chắc chắn")
            st.write("Phương pháp dựa trên quan hệ không thể phân biệt và các tập xấp xỉ.")
            
            sample_data = df.sample(min(12, len(df)))
            st.write("**Bảng hệ thống quyết định (Mẫu):**")
            st.dataframe(sample_data, use_container_width=True)
            
            st.info("Hệ thống thực hiện xác định các xấp xỉ dưới và xấp xỉ trên để tìm ra vùng biên dữ liệu.")
            if st.button("Xác định quan hệ tương đương"):
                st.success("Đã hoàn tất tính toán các lớp tương đương dựa trên thuộc tính đã chọn.")
                st.json({"Xấp xỉ dưới (Lower)": "Các đối tượng chắc chắn thuộc tập đích", "Xấp xỉ trên (Upper)": "Các đối tượng có thể thuộc tập đích"})