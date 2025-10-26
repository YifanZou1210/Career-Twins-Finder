# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.feature_engine import FeatureEngine
from src.matcher import CareerMatcher
from src.utils import load_profiles, format_salary

# 页面配置
st.set_page_config(
    page_title="Career Twin Finder",
    page_icon="👥",
    layout="wide"
)

# 缓存数据加载
@st.cache_data
def load_and_process_data():
    """加载和处理数据"""

    # 尝试加载科技行业数据，如果不存在则加载通用数据
    try:
        profiles = load_profiles('processed_data/tech_profiles.pkl')
    except:
        profiles = load_profiles('processed_data/profiles.pkl')

    # 特征工程
    feature_engine = FeatureEngine()
    feature_engine.fit(profiles)
    feature_matrix = feature_engine.transform(profiles)

    # 创建matcher
    matcher = CareerMatcher(profiles, feature_matrix)

    return profiles, matcher, feature_engine

# 加载数据
try:
    profiles, matcher, feature_engine = load_and_process_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run: python prepare_data.py")
    st.stop()

# 标题
st.title("👥 Career Twin Finder")
st.markdown("Find tech professionals with similar career paths and get personalized skill recommendations!")

# 侧边栏
with st.sidebar:
    st.header("Select a Profile")

    # 搜索过滤
    search_term = st.text_input("Search by role or company")

    # 过滤profiles
    if search_term:
        filtered_indices = [
            i for i, p in enumerate(profiles)
            if search_term.lower() in p['current_role'].lower() or
               search_term.lower() in p['current_company'].lower()
        ][:100]
    else:
        filtered_indices = list(range(min(100, len(profiles))))

    # 选择profile
    profile_options = [
        f"{profiles[i]['current_role']} @ {profiles[i]['current_company']}"
        for i in filtered_indices
    ]

    if profile_options:
        selected_option = st.selectbox("Choose a profile:", profile_options)
        selected_idx = filtered_indices[profile_options.index(selected_option)]
        selected_profile = profiles[selected_idx]
    else:
        st.error("No profiles found")
        st.stop()

    # 显示选中的profile
    st.subheader("Profile Details")
    st.write(f"**Role:** {selected_profile['current_role']}")
    st.write(f"**Company:** {selected_profile['current_company']}")
    st.write(f"**Location:** {selected_profile['location']}")

    # 显示级别
    level_names = {
        1: 'Junior (0-2 years)',
        2: 'Mid-level (2-5 years)',
        3: 'Senior (5-8 years)',
        4: 'Staff/Principal (8+ years)',
        5: 'Management'
    }
    st.write(f"**Level:** {level_names.get(selected_profile['experience_level'], 'Unknown')}")

    # 显示专业领域（如果有）
    if selected_profile.get('specialization'):
        st.write(f"**Specialization:** {selected_profile['specialization']}")

    # 显示技能
    if selected_profile.get('skills'):
        st.write("**Tech Stack:**")
        for skill in selected_profile['skills'][:6]:
            st.write(f"• {skill}")

    # 参数
    st.subheader("Settings")
    n_twins = st.slider("Number of twins:", 3, 10, 5)

# 主要内容
tab1, tab2, tab3 = st.tabs(["👥 Career Twins", "🎯 Skill Recommendations", "📈 Career Predictions"])

# 找到twins（所有标签页都需要）
twins = matcher.find_twins(selected_idx, n_twins)

with tab1:
    st.header("Your Career Twins in Tech")

    for i, twin in enumerate(twins, 1):
        similarity_pct = twin['similarity'] * 100

        with st.expander(f"Twin #{i} - {similarity_pct:.1f}% Match"):
            twin_profile = twin['profile']

            # 基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Role:** {twin_profile['current_role']}")
                st.write(f"**Company:** {twin_profile['current_company']}")
                st.write(f"**Company Type:** {twin_profile.get('company_type', 'Unknown')}")
                st.write(f"**Location:** {twin_profile['location']}")

            with col2:
                st.write(f"**Level:** {level_names.get(twin_profile['experience_level'], 'Unknown')}")
                st.write(f"**Specialization:** {twin_profile.get('specialization', 'General')}")
                st.write(f"**Work Type:** {twin_profile['work_type']}")
                if twin_profile.get('salary_info'):
                    st.write(f"**Salary:** {format_salary(twin_profile['salary_info'])}")

            # 技术栈
            if twin_profile.get('skills'):
                st.write("**Tech Stack:**")
                skills_str = ', '.join(twin_profile['skills'][:8])
                st.write(skills_str)

            # 相似原因（使用新方法）
            explanations, match_strength = matcher.explain_match_detailed(selected_idx, twin['index'])
            if explanations:
                st.write("**Why Similar:**")
                for explanation in explanations:
                    if 'Strong' in explanation or 'Same specialization' in explanation:
                        st.write(f"🔥 {explanation}")
                    else:
                        st.write(f"✓ {explanation}")

                # 匹配强度指示
                if match_strength >= 5:
                    st.success("Very Strong Match!")
                elif match_strength >= 3:
                    st.info("Good Match")

with tab2:
    st.header("🎯 Personalized Skill Recommendations")
    st.markdown("Based on your career twins, here are the skills you should consider learning:")

    # 获取技能建议
    skill_recommendations = matcher.get_skill_recommendations(selected_idx, twins)

    if skill_recommendations:
        for i, rec in enumerate(skill_recommendations, 1):
            col1, col2 = st.columns([3, 1])

            with col1:
                # 技能名称和类型
                st.subheader(f"{i}. {rec['skill']}")
                st.write(f"**Type:** {rec['type']}")
                st.write(f"**Why learn this:** {rec['reason']}")

                # 进度条显示采用率
                adoption_rate = rec['adoption_rate']
                st.progress(adoption_rate / 100)
                st.write(f"Adoption rate among similar professionals: {adoption_rate:.0f}%")

            with col2:
                # 优先级指示
                priority = rec['priority']
                if priority > 80:
                    st.metric("Priority", "HIGH", delta="Learn Now")
                elif priority > 50:
                    st.metric("Priority", "MEDIUM", delta="Plan Soon")
                else:
                    st.metric("Priority", "LOW", delta="Consider")

        # 技能分布图表
        st.subheader("Skills Landscape")

        skill_names = [r['skill'] for r in skill_recommendations]
        adoption_rates = [r['adoption_rate'] for r in skill_recommendations]

        fig = px.bar(
            x=adoption_rates,
            y=skill_names,
            orientation='h',
            labels={'x': 'Adoption Rate (%)', 'y': 'Technology'},
            title="Technology Adoption Among Your Career Twins",
            color=adoption_rates,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("You seem to have all the key skills for your level! Consider deepening expertise in your current stack.")

with tab3:
    st.header("📈 Predicted Career Paths")
    st.markdown("Based on professionals similar to you, here are potential next steps:")

    # 获取职业预测
    predictions = matcher.predict_next_roles(selected_idx, twins)

    if predictions:
        # 饼图显示概率
        roles = [p['role'] for p in predictions]
        probs = [p['probability'] for p in predictions]

        fig = px.pie(
            values=probs,
            names=roles,
            title="Potential Next Roles",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        # 详细信息
        for pred in predictions:
            with st.expander(f"📊 {pred['role']} - {pred['probability']:.1f}% likelihood"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Confidence:** {pred['confidence']:.0f}%")
                    st.write(f"**Based on:** {pred['based_on_count']} similar professionals")

                    if pred.get('typical_companies'):
                        st.write(f"**Typical companies:** {', '.join(pred['typical_companies'])}")

                with col2:
                    if pred.get('required_skills'):
                        st.write("**Key skills needed:**")
                        for skill in pred['required_skills']:
                            st.write(f"• {skill}")
    else:
        st.info("Need more data to predict career paths. Try selecting a different profile or adjusting the number of twins.")

# 底部统计
st.header("📊 Dataset Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Tech Profiles", len(profiles))

with col2:
    tech_companies = len(set(p['current_company'] for p in profiles if p.get('industry') == 'Technology'))
    st.metric("Tech Companies", tech_companies)

with col3:
    unique_skills = set()
    for p in profiles:
        unique_skills.update(p.get('skills', []))
    st.metric("Technologies Tracked", len(unique_skills))

with col4:
    avg_skills = sum(len(p.get('skills', [])) for p in profiles) / len(profiles)
    st.metric("Avg Tech Stack Size", f"{avg_skills:.1f}")