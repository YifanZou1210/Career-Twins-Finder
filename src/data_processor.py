# src/data_processor.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

class DataProcessor:
    def __init__(self, data_path='data/linkedin-job-postings'):
        self.data_path = Path(data_path)
        self.postings = None

    def load_data(self):
        """加载LinkedIn postings数据"""
        csv_path = self.data_path / 'postings.csv'
        print(f"Loading data from {csv_path}")

        try:
            self.postings = pd.read_csv(csv_path,
                                        low_memory=False,
                                        on_bad_lines='skip')
            print(f"Loaded {len(self.postings)} job postings")

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

        return self.postings

    def create_tech_profiles(self, sample_size=5000):
        """只创建科技行业的profiles"""

        if self.postings is None:
            self.load_data()

        df = self.postings.copy()
        df = df.dropna(subset=['title', 'company_name'])

        # 筛选科技相关职位
        tech_filtered = []
        for idx, row in df.iterrows():
            if self._is_tech_job(row):
                tech_filtered.append(row)
            if len(tech_filtered) >= sample_size:
                break

        tech_df = pd.DataFrame(tech_filtered)
        print(f"Found {len(tech_df)} tech jobs")

        profiles = []
        for idx, row in tech_df.iterrows():
            profile = self._create_tech_profile(row, idx)
            if profile:
                profiles.append(profile)

        print(f"Created {len(profiles)} tech profiles")
        return profiles

    def _is_tech_job(self, row):
        """判断是否为科技行业职位"""

        text = f"{row.get('title', '')} {row.get('company_name', '')} {str(row.get('description', ''))[:500]}".lower()

        # 科技职位关键词
        tech_indicators = [
            'software', 'engineer', 'developer', 'programmer',
            'frontend', 'backend', 'fullstack', 'full-stack', 'full stack',
            'devops', 'cloud', 'database', 'api', 'web',
            'python', 'java', 'javascript', 'react', 'node',
            'data', 'machine learning', 'ai', 'analytics',
            'technical', 'coding', 'programming', 'development',
            'architect', 'system', 'network', 'security',
            'qa', 'quality assurance', 'test', 'automation'
        ]

        # 至少匹配3个关键词
        matches = sum(1 for keyword in tech_indicators if keyword in text)
        return matches >= 2

    def _create_tech_profile(self, row, idx):
        """创建科技行业profile"""

        title = self._standardize_tech_title(row.get('title', 'Unknown'))
        company = row.get('company_name', 'Unknown')
        location = row.get('location', 'Unknown')

        # 提取技术栈
        tech_stack = self._extract_tech_skills(
            str(row.get('skills_desc', '')) + ' ' +
            str(row.get('description', ''))[:1000]
        )

        # 标准化经验级别
        exp_level = self._extract_tech_experience_level(
            title,
            row.get('formatted_experience_level', ''),
            str(row.get('description', ''))
        )

        # 生成科技职业路径
        career_path = self._generate_tech_career_path(title, exp_level)

        # 提取薪资
        salary_info = self._extract_salary(row)

        profile = {
            'profile_id': f'tech_profile_{idx:06d}',
            'current_role': title,
            'current_company': company,
            'location': location,
            'skills': tech_stack,
            'experience_level': exp_level,
            'career_path': career_path,
            'salary_info': salary_info,
            'work_type': row.get('formatted_work_type', 'Full-time'),
            'remote_allowed': row.get('remote_allowed', 0),
            'industry': 'Technology',  # 统一为科技行业
            'specialization': self._get_tech_specialization(title, tech_stack),
            'company_type': self._classify_tech_company(company)
        }

        return profile

    def _standardize_tech_title(self, title):
        """标准化科技职位名称"""
        if pd.isna(title):
            return "Software Engineer"

        title = str(title).strip()

        # 标准化映射
        title_mapping = {
            'swe': 'Software Engineer',
            'sw engineer': 'Software Engineer',
            'programmer': 'Software Developer',
            'coder': 'Software Developer',
            'front end': 'Frontend',
            'back end': 'Backend',
            'full-stack': 'Full Stack',
            'sr': 'Senior',
            'jr': 'Junior',
            'dev': 'Developer',
            'eng': 'Engineer'
        }

        title_lower = title.lower()
        for old, new in title_mapping.items():
            title_lower = title_lower.replace(old, new.lower())

        # 规范化格式
        return ' '.join(word.capitalize() for word in title_lower.split())

    def _extract_tech_skills(self, text):
        """提取技术栈技能"""
        if pd.isna(text):
            return []

        text = str(text).upper()

        # 技术技能分类
        tech_skills = {
            'languages': [
                'PYTHON', 'JAVA', 'JAVASCRIPT', 'TYPESCRIPT', 'C\+\+', 'C#',
                'GO', 'RUST', 'KOTLIN', 'SWIFT', 'RUBY', 'PHP', 'SCALA'
            ],
            'frontend': [
                'REACT', 'ANGULAR', 'VUE', 'NEXT\.?JS', 'NUXT', 'SVELTE',
                'HTML', 'CSS', 'SASS', 'WEBPACK', 'REDUX', 'MOBX'
            ],
            'backend': [
                'NODE\.?JS', 'EXPRESS', 'DJANGO', 'FLASK', 'SPRING',
                'FASTAPI', 'RAILS', 'LARAVEL', 'NET CORE', 'GRAPHQL'
            ],
            'database': [
                'SQL', 'POSTGRESQL', 'MYSQL', 'MONGODB', 'REDIS',
                'ELASTICSEARCH', 'CASSANDRA', 'DYNAMODB', 'ORACLE'
            ],
            'cloud': [
                'AWS', 'AZURE', 'GCP', 'DOCKER', 'KUBERNETES',
                'TERRAFORM', 'CI/CD', 'JENKINS', 'GITLAB'
            ],
            'data': [
                'MACHINE LEARNING', 'TENSORFLOW', 'PYTORCH', 'SCIKIT-LEARN',
                'PANDAS', 'NUMPY', 'SPARK', 'HADOOP', 'TABLEAU'
            ]
        }

        found_skills = []
        for category, skills_list in tech_skills.items():
            for skill in skills_list:
                if re.search(skill, text):
                    clean_skill = skill.replace('\\', '').replace('.?', '')
                    found_skills.append(clean_skill.title())

        return list(set(found_skills))[:10]  # 最多10个技能

    def _extract_tech_experience_level(self, title, exp_str, description):
        """科技行业经验级别判断"""
        text = f"{title} {exp_str} {description[:500]}".lower()

        # 科技行业的级别判断
        if any(word in text for word in ['intern', 'entry', 'junior', 'associate']):
            return 1  # Junior
        elif any(word in text for word in ['mid-level', 'mid level', '2-5 years', '3-5 years']):
            return 2  # Mid
        elif any(word in text for word in ['senior', 'lead', 'sr.', '5+ years', '7+ years']):
            return 3  # Senior
        elif any(word in text for word in ['staff', 'principal', 'architect']):
            return 4  # Staff/Principal
        elif any(word in text for word in ['manager', 'director', 'vp', 'cto']):
            return 5  # Management
        else:
            # 默认根据年限判断
            if '0-2' in text or '1-3' in text:
                return 1
            elif '10+' in text or '15+' in text:
                return 4
            return 2

    def _generate_tech_career_path(self, current_title, exp_level):
        """生成科技行业典型职业路径"""

        tech_paths = {
            1: [  # Junior path
                {'role': 'Intern/Entry Level', 'duration': 1},
                {'role': current_title, 'duration': 1}
            ],
            2: [  # Mid path
                {'role': 'Junior Developer', 'duration': 2},
                {'role': current_title, 'duration': 2}
            ],
            3: [  # Senior path
                {'role': 'Junior Developer', 'duration': 2},
                {'role': 'Software Engineer', 'duration': 3},
                {'role': current_title, 'duration': 2}
            ],
            4: [  # Staff/Principal path
                {'role': 'Software Engineer', 'duration': 3},
                {'role': 'Senior Software Engineer', 'duration': 3},
                {'role': current_title, 'duration': 2}
            ],
            5: [  # Management path
                {'role': 'Senior Engineer', 'duration': 4},
                {'role': 'Tech Lead', 'duration': 3},
                {'role': current_title, 'duration': 2}
            ]
        }

        return tech_paths.get(exp_level, tech_paths[2])

    def _get_tech_specialization(self, title, skills):
        """判断技术专长领域"""
        title_lower = title.lower()
        skills_text = ' '.join(skills).lower()

        if any(word in title_lower for word in ['frontend', 'ui', 'ux']):
            return 'Frontend'
        elif any(word in title_lower for word in ['backend', 'api', 'server']):
            return 'Backend'
        elif any(word in title_lower for word in ['full stack', 'fullstack']):
            return 'Full Stack'
        elif any(word in title_lower for word in ['data', 'ml', 'ai', 'machine learning']):
            return 'Data/ML'
        elif any(word in title_lower for word in ['devops', 'sre', 'infrastructure']):
            return 'DevOps/Infrastructure'
        elif any(word in title_lower for word in ['mobile', 'ios', 'android']):
            return 'Mobile'
        else:
            # 基于技能判断
            if 'react' in skills_text or 'vue' in skills_text:
                return 'Frontend'
            elif 'django' in skills_text or 'spring' in skills_text:
                return 'Backend'
            return 'General Software'

    def _classify_tech_company(self, company_name):
        """分类科技公司类型"""
        if pd.isna(company_name):
            return 'Unknown'

        company_lower = company_name.lower()

        # FAANG/大厂
        if any(c in company_lower for c in ['google', 'amazon', 'meta', 'facebook', 'apple', 'microsoft', 'netflix']):
            return 'FAANG'
        # 独角兽
        elif any(c in company_lower for c in ['uber', 'airbnb', 'stripe', 'databricks', 'spacex']):
            return 'Unicorn'
        # 传统科技公司
        elif any(c in company_lower for c in ['ibm', 'oracle', 'cisco', 'intel', 'hp']):
            return 'Enterprise'
        # 创业公司标志
        elif 'startup' in company_lower or 'labs' in company_lower:
            return 'Startup'
        else:
            return 'Mid-size Tech'

    def _extract_salary(self, row):
        """提取薪资信息"""
        salary_info = {}

        if pd.notna(row.get('min_salary')):
            salary_info['min'] = float(row['min_salary'])
        if pd.notna(row.get('max_salary')):
            salary_info['max'] = float(row['max_salary'])
        if pd.notna(row.get('med_salary')):
            salary_info['median'] = float(row['med_salary'])

        # 转换为年薪
        if row.get('pay_period') == 'HOURLY' and salary_info:
            for key in salary_info:
                salary_info[key] = salary_info[key] * 2080

        return salary_info if salary_info else None