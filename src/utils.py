import json
import pickle
import numpy as np  # 添加这行
from pathlib import Path

# ... 其余代码不变 ...

def save_profiles(profiles, filepath):
    """保存profiles到文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(profiles, f)

    print(f"Saved {len(profiles)} profiles to {filepath}")

def load_profiles(filepath):
    """从文件加载profiles"""
    with open(filepath, 'rb') as f:
        profiles = pickle.load(f)

    print(f"Loaded {len(profiles)} profiles from {filepath}")
    return profiles

def get_profile_stats(profiles):
    """获取profiles统计信息"""
    stats = {
        'total_profiles': len(profiles),
        'unique_companies': len(set(p['current_company'] for p in profiles)),
        'unique_roles': len(set(p['current_role'] for p in profiles)),
        'unique_locations': len(set(p['location'] for p in profiles)),
        'avg_skills': np.mean([len(p.get('skills', [])) for p in profiles]),
        'profiles_with_salary': sum(1 for p in profiles if p.get('salary_info'))
    }
    return stats

def format_salary(salary_info):
    """格式化薪资显示"""
    if not salary_info:
        return "Not Available"

    if 'median' in salary_info:
        median = salary_info['median']
        if median > 1000:
            return f"${median/1000:.0f}k/year"
        else:
            return f"${median:.0f}/hour"

    return "Not Available"