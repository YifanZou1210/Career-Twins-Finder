# src/feature_engine.py
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

class FeatureEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.role_encoder = LabelEncoder()
        self.company_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.specialization_encoder = LabelEncoder()
        self.company_type_encoder = LabelEncoder()
        self.all_skills = []

    def fit(self, profiles):
        """训练特征转换器"""

        # 收集所有唯一值
        roles = [p.get('current_role', 'Unknown') for p in profiles] + ['Unknown']
        companies = [p.get('current_company', 'Unknown') for p in profiles] + ['Unknown']
        locations = [p.get('location', 'Unknown') for p in profiles] + ['Unknown']
        specializations = [p.get('specialization', 'General') for p in profiles] + ['General', 'Unknown']
        company_types = [p.get('company_type', 'Unknown') for p in profiles] + ['Unknown']

        # 训练编码器
        self.role_encoder.fit(roles)
        self.company_encoder.fit(companies)
        self.location_encoder.fit(locations)
        self.specialization_encoder.fit(specializations)
        self.company_type_encoder.fit(company_types)

        # 收集所有技能并计数
        all_skills_list = []
        for p in profiles:
            all_skills_list.extend(p.get('skills', []))

        # 获取最常见的30个技能（科技行业技能更多）
        skill_counts = Counter(all_skills_list)
        self.all_skills = [skill for skill, _ in skill_counts.most_common(30)]

        print(f"Feature Engine initialized with {len(self.all_skills)} skills")

        return self

    def transform(self, profiles):
        """将profiles转换为特征矩阵"""

        features = []
        for profile in profiles:
            feature_vector = self._extract_features(profile)
            features.append(feature_vector)

        features_array = np.array(features)

        # 处理NaN值
        features_array = np.nan_to_num(features_array, nan=0.0)

        # 标准化数值特征
        features_normalized = self.scaler.fit_transform(features_array)

        # 再次确保没有NaN
        features_normalized = np.nan_to_num(features_normalized, nan=0.0)

        return features_normalized

    def _extract_features(self, profile):
        """提取单个profile的特征"""

        features = []

        # === 基础特征 ===

        # 1. 经验级别 (1-5)
        features.append(float(profile.get('experience_level', 2)))

        # 2. 技能数量
        features.append(float(len(profile.get('skills', []))))

        # 3. 职业路径长度
        features.append(float(len(profile.get('career_path', []))))

        # 4. 总工作年限
        career_path = profile.get('career_path', [])
        total_years = sum(p.get('duration', 0) for p in career_path) if career_path else 0
        features.append(float(total_years))

        # 5. 平均任职时长
        avg_duration = total_years / len(career_path) if career_path else 2.0
        features.append(float(avg_duration))

        # 6. 远程工作
        remote = profile.get('remote_allowed', 0)
        features.append(float(remote) if remote is not None else 0.0)

        # 7. 薪资特征（标准化到0-10范围）
        salary = profile.get('salary_info', {})
        if salary and 'median' in salary and salary['median'] is not None:
            # 将薪资映射到0-10的范围
            salary_val = min(float(salary['median']), 500000)  # cap at 500k
            normalized_salary = (salary_val / 500000) * 10
            features.append(normalized_salary)
        else:
            features.append(5.0)  # 默认中位数

        # === 分类特征编码 ===

        # 8. 工作类型
        work_type_map = {
            'Full-time': 1.0,
            'Part-time': 0.5,
            'Contract': 0.7,
            'Internship': 0.2,
            'Freelance': 0.6
        }
        work_type = profile.get('work_type', 'Full-time')
        features.append(work_type_map.get(work_type, 1.0))

        # 9. 专业领域编码
        try:
            spec = profile.get('specialization', 'General')
            if spec in self.specialization_encoder.classes_:
                spec_encoded = float(self.specialization_encoder.transform([spec])[0])
            else:
                spec_encoded = float(self.specialization_encoder.transform(['General'])[0])
        except:
            spec_encoded = 0.0
        features.append(spec_encoded)

        # 10. 公司类型编码
        try:
            comp_type = profile.get('company_type', 'Unknown')
            if comp_type in self.company_type_encoder.classes_:
                comp_type_encoded = float(self.company_type_encoder.transform([comp_type])[0])
            else:
                comp_type_encoded = float(self.company_type_encoder.transform(['Unknown'])[0])
        except:
            comp_type_encoded = 0.0
        features.append(comp_type_encoded)

        # 11. 职位编码
        try:
            role = profile.get('current_role', 'Unknown')
            if role in self.role_encoder.classes_:
                role_encoded = float(self.role_encoder.transform([role])[0])
            else:
                role_encoded = float(self.role_encoder.transform(['Unknown'])[0])
        except:
            role_encoded = 0.0
        features.append(role_encoded / len(self.role_encoder.classes_))  # 归一化

        # 12. 公司编码
        try:
            company = profile.get('current_company', 'Unknown')
            if company in self.company_encoder.classes_:
                company_encoded = float(self.company_encoder.transform([company])[0])
            else:
                company_encoded = float(self.company_encoder.transform(['Unknown'])[0])
        except:
            company_encoded = 0.0
        features.append(company_encoded / len(self.company_encoder.classes_))  # 归一化

        # 13. 地点编码
        try:
            location = profile.get('location', 'Unknown')
            if location in self.location_encoder.classes_:
                location_encoded = float(self.location_encoder.transform([location])[0])
            else:
                location_encoded = float(self.location_encoder.transform(['Unknown'])[0])
        except:
            location_encoded = 0.0
        features.append(location_encoded / len(self.location_encoder.classes_))  # 归一化

        # === 技能特征 (One-hot encoding for top 30 skills) ===

        profile_skills = set(profile.get('skills', []))

        # 14-43. 技能one-hot编码（30个技能）
        for skill in self.all_skills:
            features.append(1.0 if skill in profile_skills else 0.0)

        # === 技能类别特征 ===

        # 44. 编程语言数量
        prog_langs = ['Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'Go', 'Rust', 'Ruby']
        prog_count = sum(1 for lang in prog_langs if lang in profile_skills)
        features.append(float(prog_count) / 5.0)  # 归一化

        # 45. 前端技能数量
        frontend_skills = ['React', 'Angular', 'Vue', 'HTML', 'CSS', 'Redux']
        frontend_count = sum(1 for skill in frontend_skills if skill in profile_skills)
        features.append(float(frontend_count) / 3.0)  # 归一化

        # 46. 后端技能数量
        backend_skills = ['Node.Js', 'Django', 'Flask', 'Spring', 'Express']
        backend_count = sum(1 for skill in backend_skills if skill in profile_skills)
        features.append(float(backend_count) / 3.0)  # 归一化

        # 47. 云技术技能数量
        cloud_skills = ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes']
        cloud_count = sum(1 for skill in cloud_skills if skill in profile_skills)
        features.append(float(cloud_count) / 3.0)  # 归一化

        # 48. 数据库技能数量
        db_skills = ['SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis']
        db_count = sum(1 for skill in db_skills if skill in profile_skills)
        features.append(float(db_count) / 3.0)  # 归一化

        # 49. AI/ML技能数量
        ml_skills = ['Machine Learning', 'TensorFlow', 'PyTorch', 'Scikit-Learn']
        ml_count = sum(1 for skill in ml_skills if skill in profile_skills)
        features.append(float(ml_count) / 2.0)  # 归一化

        # 50. 技能多样性指数（0-1）
        skill_diversity = len(set([
            'languages' if prog_count > 0 else None,
            'frontend' if frontend_count > 0 else None,
            'backend' if backend_count > 0 else None,
            'cloud' if cloud_count > 0 else None,
            'database' if db_count > 0 else None,
            'ml' if ml_count > 0 else None
        ]) - {None}) / 6.0
        features.append(skill_diversity)

        # 确保特征长度一致（填充到50个特征）
        while len(features) < 50:
            features.append(0.0)

        # 确保所有值都是float且不是NaN
        features = [float(f) if f is not None else 0.0 for f in features[:50]]

        return features

    def get_feature_importance(self):
        """返回特征名称列表，用于分析"""
        feature_names = [
            'experience_level',
            'num_skills',
            'career_path_length',
            'total_years',
            'avg_tenure',
            'remote_allowed',
            'salary_normalized',
            'work_type',
            'specialization',
            'company_type',
            'role_encoded',
            'company_encoded',
            'location_encoded'
        ]

        # 添加技能名称
        for skill in self.all_skills:
            feature_names.append(f'skill_{skill}')

        # 添加技能类别
        feature_names.extend([
            'prog_lang_count',
            'frontend_count',
            'backend_count',
            'cloud_count',
            'database_count',
            'ml_count',
            'skill_diversity'
        ])

        return feature_names[:50]