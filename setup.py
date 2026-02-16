from setuptools import setup, find_packages

setup(
    name="nova-cli",
    version="0.1.0",
    description="A powerful command-line interface tool named Nova",
    author="Your Name",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nova=nova.main:main',
        ],
    },
    python_requires='>=3.6',
    install_requires=[
        'rich',
        'ollama',
        'psutil',
        'requests',
        'numpy',
        'huggingface_hub',
        'pyautogui',
        'screen_brightness_control',
        'pycaw; platform_system=="Windows"',
        'comtypes; platform_system=="Windows"',
    ],
)
