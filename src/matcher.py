# src/matcher.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class CareerMatcher:
    def __init__(self, profiles, feature_matrix):
        self.profiles = profiles
        self.feature_matrix = feature_matrix
        self.similarity_matrix = None

    def compute_similarities(self):
        """计算所有profiles之间的相似度"""
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        return self.similarity_matrix

    def find_twins(self, profile_idx, n_twins=5):
        """找到最相似的profiles"""

        if self.similarity_matrix is None:
            self.compute_similarities()

        similarities = self.similarity_matrix[profile_idx]
        indices = np.argsort(similarities)[::-1]

        # 排除自己
        twin_indices = [i for i in indices if i != profile_idx][:n_twins]

        twins = []
        for idx in twin_indices:
            twins.append({
                'profile': self.profiles[idx],
                'similarity': similarities[idx],
                'index': idx
            })

        return twins

    def explain_match_detailed(self, profile1_idx, profile2_idx):
        """提供详细的匹配解释"""

        p1 = self.profiles[profile1_idx]
        p2 = self.profiles[profile2_idx]

        explanations = []
        match_strength = 0

        # 1. 技术栈匹配分析
        skills1 = set(p1.get('skills', []))
        skills2 = set(p2.get('skills', []))
        common_skills = skills1.intersection(skills2)

        if len(common_skills) >= 4:
            explanations.append(f"Strong tech stack match: {', '.join(list(common_skills)[:4])}")
            match_strength += 3
        elif len(common_skills) >= 2:
            explanations.append(f"Shared technologies: {', '.join(common_skills)}")
            match_strength += 2

        # 2. 专业领域匹配
        if p1.get('specialization') == p2.get('specialization'):
            explanations.append(f"Same specialization: {p1['specialization']}")
            match_strength += 2

        # 3. 职位级别和经验
        if p1['experience_level'] == p2['experience_level']:
            level_names = {
                1: 'Junior (0-2 years)',
                2: 'Mid-level (2-5 years)',
                3: 'Senior (5-8 years)',
                4: 'Staff/Principal (8+ years)',
                5: 'Management/Leadership'
            }
            explanations.append(f"Same level: {level_names.get(p1['experience_level'])}")
            match_strength += 1

        # 4. 公司类型
        if p1.get('company_type') == p2.get('company_type'):
            explanations.append(f"Similar company type: {p1['company_type']}")
            match_strength += 1

        # 5. 工作模式
        if p1.get('remote_allowed') == p2.get('remote_allowed') and p1.get('remote_allowed'):
            explanations.append("Both remote-friendly positions")

        return explanations, match_strength

    def get_skill_recommendations(self, profile_idx, twins):
        """生成个性化技能建议"""

        current_profile = self.profiles[profile_idx]
        current_skills = set(current_profile.get('skills', []))
        current_level = current_profile['experience_level']
        current_spec = current_profile.get('specialization', '')

        skill_analysis = {}

        for twin in twins:
            twin_profile = twin['profile']
            twin_level = twin_profile['experience_level']
            twin_skills = set(twin_profile.get('skills', []))
            similarity = twin['similarity']

            # 分析更高级别的人的技能
            if twin_level >= current_level:
                missing_skills = twin_skills - current_skills

                for skill in missing_skills:
                    if skill not in skill_analysis:
                        skill_analysis[skill] = {
                            'frequency': 0,
                            'avg_level': 0,
                            'total_similarity': 0,
                            'appears_in_levels': [],
                            'specializations': []
                        }

                    skill_analysis[skill]['frequency'] += 1
                    skill_analysis[skill]['avg_level'] += twin_level
                    skill_analysis[skill]['total_similarity'] += similarity
                    skill_analysis[skill]['appears_in_levels'].append(twin_level)
                    skill_analysis[skill]['specializations'].append(
                        twin_profile.get('specialization', '')
                    )

        # 生成推荐
        recommendations = []
        for skill, data in skill_analysis.items():
            if data['frequency'] > 0:
                avg_level = data['avg_level'] / data['frequency']
                importance_score = (data['frequency'] / len(twins)) * data['total_similarity']

                # 判断技能类型和推荐理由
                skill_type = self._categorize_skill(skill)
                reason = self._generate_skill_reason(
                    skill,
                    data['frequency'],
                    len(twins),
                    avg_level,
                    current_level,
                    skill_type
                )

                recommendations.append({
                    'skill': skill,
                    'type': skill_type,
                    'importance_score': importance_score,
                    'adoption_rate': (data['frequency'] / len(twins)) * 100,
                    'typical_level': avg_level,
                    'reason': reason,
                    'priority': self._calculate_priority(
                        importance_score,
                        data['frequency'],
                        skill_type,
                        current_spec
                    )
                })

        # 排序：按优先级
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        # 返回前6个最重要的技能
        return recommendations[:6]

    def _categorize_skill(self, skill):
        """分类技能类型"""
        skill_lower = skill.lower()

        if any(lang in skill_lower for lang in ['python', 'java', 'javascript', 'typescript', 'go', 'rust']):
            return 'Programming Language'
        elif any(fw in skill_lower for fw in ['react', 'angular', 'vue', 'django', 'spring', 'express']):
            return 'Framework'
        elif any(db in skill_lower for db in ['sql', 'mongodb', 'redis', 'postgresql']):
            return 'Database'
        elif any(cloud in skill_lower for cloud in ['aws', 'azure', 'gcp', 'docker', 'kubernetes']):
            return 'Cloud/DevOps'
        elif any(ml in skill_lower for ml in ['machine learning', 'tensorflow', 'pytorch']):
            return 'AI/ML'
        else:
            return 'Tool/Technology'

    def _generate_skill_reason(self, skill, frequency, total_twins, avg_level, current_level, skill_type):
        """生成学习技能的理由"""

        adoption_rate = (frequency / total_twins) * 100

        if adoption_rate >= 60:
            return f"Essential skill - {adoption_rate:.0f}% of similar professionals have mastered this"
        elif avg_level > current_level and adoption_rate >= 40:
            return f"Key for advancement - common at level {avg_level:.1f} ({adoption_rate:.0f}% adoption)"
        elif skill_type == 'Cloud/DevOps' and adoption_rate >= 30:
            return "Critical cloud skill for modern tech stack"
        elif skill_type == 'AI/ML':
            return "High-growth area with increasing demand"
        elif adoption_rate >= 30:
            return f"Increasingly important - {adoption_rate:.0f}% adoption among peers"
        else:
            return "Emerging technology to differentiate yourself"

    def _calculate_priority(self, importance_score, frequency, skill_type, current_spec):
        """计算技能学习优先级"""

        priority = importance_score * 100

        # 根据技能类型调整
        if skill_type == 'Cloud/DevOps':
            priority *= 1.2  # 云技术加权
        elif skill_type == 'AI/ML':
            priority *= 1.15  # AI/ML加权

        # 根据频率调整
        if frequency >= 3:
            priority *= 1.1

        return priority

    def predict_next_roles(self, profile_idx, twins):
        """预测下一步职业发展"""

        current_profile = self.profiles[profile_idx]
        current_level = current_profile['experience_level']
        current_spec = current_profile.get('specialization', '')

        role_predictions = {}

        for twin in twins:
            twin_profile = twin['profile']
            twin_level = twin_profile['experience_level']
            twin_role = twin_profile['current_role']

            # 只看高一级的职位
            if twin_level == current_level + 1:
                if twin_role not in role_predictions:
                    role_predictions[twin_role] = {
                        'count': 0,
                        'similarity_sum': 0,
                        'required_skills': [],
                        'company_types': [],
                        'specializations': []
                    }

                role_predictions[twin_role]['count'] += 1
                role_predictions[twin_role]['similarity_sum'] += twin['similarity']
                role_predictions[twin_role]['required_skills'].extend(
                    twin_profile.get('skills', [])
                )
                role_predictions[twin_role]['company_types'].append(
                    twin_profile.get('company_type', '')
                )
                role_predictions[twin_role]['specializations'].append(
                    twin_profile.get('specialization', '')
                )

        # 生成预测结果
        predictions = []
        for role, data in role_predictions.items():
            if data['count'] > 0:
                # 分析必备技能
                skill_counter = Counter(data['required_skills'])
                top_skills = [
                    skill for skill, count in skill_counter.most_common(5)
                    if count >= data['count'] * 0.5  # 至少50%的人有这个技能
                ]

                # 分析公司类型
                company_counter = Counter(data['company_types'])
                typical_companies = [
                    comp for comp, _ in company_counter.most_common(2)
                    if comp
                ]

                probability = (data['similarity_sum'] / len(twins)) * 100

                predictions.append({
                    'role': role,
                    'probability': probability,
                    'confidence': (data['count'] / len(twins)) * 100,
                    'required_skills': top_skills[:3],
                    'typical_companies': typical_companies,
                    'based_on_count': data['count']
                })

        # 排序
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        return predictions[:3]  # 返回前3个最可能的职位