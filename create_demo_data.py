# create_demo_data.py
import pickle
import random
from pathlib import Path

def create_demo_profiles(n=500):
    """为演示创建模拟profiles"""

    profiles = []

    roles = ['Software Engineer', 'Senior Developer', 'Data Scientist',
             'Frontend Developer', 'Backend Engineer', 'DevOps Engineer',
             'Full Stack Developer', 'ML Engineer', 'Tech Lead']

    companies = ['Google', 'Amazon', 'Microsoft', 'Meta', 'Apple',
                 'Netflix', 'Uber', 'Airbnb', 'Stripe', 'Databricks']

    skills_pool = ['Python', 'Java', 'JavaScript', 'React', 'Node.js',
                   'AWS', 'Docker', 'Kubernetes', 'SQL', 'MongoDB',
                   'TypeScript', 'Go', 'Machine Learning', 'Git']

    locations = ['San Francisco, CA', 'New York, NY', 'Seattle, WA',
                 'Austin, TX', 'Boston, MA']

    for i in range(n):
        level = random.randint(1, 5)

        profile = {
            'profile_id': f'demo_{i:06d}',
            'current_role': random.choice(roles),
            'current_company': random.choice(companies),
            'location': random.choice(locations),
            'skills': random.sample(skills_pool, k=random.randint(3, 8)),
            'experience_level': level,
            'career_path': [
                {'role': 'Junior Developer', 'duration': 2},
                {'role': 'Software Engineer', 'duration': 3}
            ],
            'work_type': 'Full-time',
            'remote_allowed': random.choice([0, 1]),
            'industry': 'Technology',
            'specialization': random.choice(['Frontend', 'Backend', 'Full Stack', 'Data/ML']),
            'company_type': random.choice(['FAANG', 'Unicorn', 'Startup'])
        }

        profiles.append(profile)

    return profiles

if __name__ == "__main__":
    Path('processed_data').mkdir(exist_ok=True)
    profiles = create_demo_profiles(500)

    with open('processed_data/tech_profiles.pkl', 'wb') as f:
        pickle.dump(profiles, f)

    print(f"Created {len(profiles)} demo profiles")