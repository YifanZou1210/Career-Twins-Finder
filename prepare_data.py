# prepare_data.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_processor import DataProcessor
from src.feature_engine import FeatureEngine
from src.utils import save_profiles, get_profile_stats

def main():
    print("="*50)
    print("Career Twin Finder - Tech Industry Edition")
    print("="*50)

    # 1. 加载和处理数据
    processor = DataProcessor('data/linkedin-job-postings')
    processor.load_data()

    # 2. 创建科技行业profiles
    print("\nFiltering for tech industry positions...")
    profiles = processor.create_tech_profiles(sample_size=3000)

    if not profiles:
        print("Error: No tech profiles created")
        return

    # 3. 保存profiles
    save_profiles(profiles, 'processed_data/tech_profiles.pkl')

    # 4. 显示统计
    print("\nTech Industry Dataset Statistics:")
    print(f"Total profiles: {len(profiles)}")

    # 技能统计
    all_skills = []
    for p in profiles:
        all_skills.extend(p.get('skills', []))

    from collections import Counter
    skill_counts = Counter(all_skills)

    print("\nTop 10 Technologies:")
    for skill, count in skill_counts.most_common(10):
        print(f"  {skill}: {count} profiles")

    # 专业领域统计
    specs = Counter(p.get('specialization', 'Unknown') for p in profiles)
    print("\nSpecialization Distribution:")
    for spec, count in specs.most_common():
        print(f"  {spec}: {count} profiles")

    # 公司类型统计
    company_types = Counter(p.get('company_type', 'Unknown') for p in profiles)
    print("\nCompany Type Distribution:")
    for comp_type, count in company_types.most_common():
        print(f"  {comp_type}: {count} profiles")

if __name__ == "__main__":
    main()