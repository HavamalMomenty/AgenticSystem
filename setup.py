"""
Setup and installation script for the Property Analysis System
"""

import subprocess
import sys
import os
from pathlib import Path


def install_packages():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def setup_environment():
    """Setup environment files"""
    print("🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        # Copy template to .env
        with open(env_template, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your OpenAI API key!")
    else:
        print("ℹ️  Environment file already exists or template not found")


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "storage",      # For LlamaIndex storage
        "reports",      # For generated reports
        "logs"          # For log files
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"   Created: {dir_name}/")
    
    print("✅ Directories created successfully!")


def verify_setup():
    """Verify the setup"""
    print("🔍 Verifying setup...")
    
    # Check if required directories exist
    required_dirs = ["database", "database/Unstructered_data", "database/Input", "database/Evaluate_property_example"]
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("⚠️  Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("   Please ensure your database structure is correct.")
    else:
        print("✅ All required directories found!")
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️  .env file not found. Please create it and add your OpenAI API key.")
    else:
        print("✅ Environment file found!")


def main():
    """Main setup function"""
    print("🚀 Property Analysis System Setup")
    print("=" * 40)
    
    # Install packages
    if not install_packages():
        print("❌ Setup failed during package installation")
        return
    
    # Setup environment
    setup_environment()
    
    # Create directories
    create_directories()
    
    # Verify setup
    verify_setup()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Ensure your database structure is correct")
    print("3. Run: python main.py")
    print("\nFor advanced features, check: python advanced_features.py")


if __name__ == "__main__":
    main()
