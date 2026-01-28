#!/usr/bin/env python3
"""
Setup script for RAG system.
Helps users get started quickly by checking dependencies and configuration.
"""
import sys
import os
from pathlib import Path
import subprocess


def check_python_version():
    """Check Python version is 3.10+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_env_file():
    """Check if .env file exists and has required keys."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("   Creating from template...")
        
        template = Path(".env.template")
        if template.exists():
            import shutil
            shutil.copy(template, env_file)
            print("✅ Created .env file from template")
            print("   ⚠️  IMPORTANT: Edit .env and add your API keys!")
            return False
        else:
            print("❌ .env.template not found")
            return False
    
    # Check for API keys
    env_content = env_file.read_text()
    
    has_openai = "OPENAI_API_KEY=" in env_content
    has_anthropic = "ANTHROPIC_API_KEY=" in env_content
    
    # Check if they're not just placeholders
    has_real_openai = has_openai and "your_openai_api_key_here" not in env_content
    has_real_anthropic = has_anthropic and "your_anthropic_api_key_here" not in env_content
    
    if has_real_openai and has_real_anthropic:
        print("✅ .env file configured with API keys")
        return True
    else:
        print("⚠️  .env file exists but API keys need to be set")
        if not has_real_openai:
            print("   - Missing OPENAI_API_KEY")
        if not has_real_anthropic:
            print("   - Missing ANTHROPIC_API_KEY")
        print("   Please edit .env file and add your keys")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"   Error: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data/raw",
        "data/chromadb",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created {len(directories)} directories")
    return True


def run_tests():
    """Run basic tests to verify setup."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Try importing main modules
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.config.settings import settings
        from src.pipeline import RAGPipeline
        
        print("✅ Core modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def print_next_steps(env_configured):
    """Print next steps for the user."""
    print("\n" + "=" * 80)
    print("SETUP SUMMARY")
    print("=" * 80)
    
    if env_configured:
        print("\n✅ Setup complete! You're ready to use the RAG system.")
        print("\n📚 Next steps:")
        print("   1. Run the example: python example_usage.py")
        print("   2. Read the README.md for detailed usage")
        print("   3. Check out src/pipeline.py for the main API")
    else:
        print("\n⚠️  Setup incomplete - API keys needed")
        print("\n📚 Next steps:")
        print("   1. Edit .env file and add your API keys:")
        print("      - OPENAI_API_KEY=sk-...")
        print("      - ANTHROPIC_API_KEY=sk-ant-...")
        print("   2. Run this setup script again: python setup.py")
        print("   3. Then run the example: python example_usage.py")
    
    print("\n💡 Tips:")
    print("   - Get OpenAI API key: https://platform.openai.com/api-keys")
    print("   - Get Anthropic API key: https://console.anthropic.com/")
    print("   - Documentation: See README.md")
    print("   - Questions: Check the Common Issues section in README")
    print()


def main():
    """Run setup process."""
    print("=" * 80)
    print("RAG SYSTEM SETUP")
    print("=" * 80)
    print()
    
    # Run checks
    checks = []
    
    print("🔍 Checking prerequisites...")
    print("-" * 80)
    checks.append(check_python_version())
    
    print("\n🔍 Checking configuration...")
    print("-" * 80)
    env_configured = check_env_file()
    checks.append(env_configured)
    
    # Create directories
    checks.append(create_directories())
    
    # Install dependencies
    install_choice = input("\n📦 Install/update dependencies? (y/n): ").lower().strip()
    if install_choice == 'y':
        checks.append(install_dependencies())
    else:
        print("⏭️  Skipping dependency installation")
    
    # Run tests
    checks.append(run_tests())
    
    # Print summary
    print_next_steps(env_configured)
    
    # Exit code
    if all(checks):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
