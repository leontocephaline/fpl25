# FPL Weekly Updater - Executable Version

## 🚀 Quick Start for End Users

### What is this?
This is a standalone executable that analyzes Fantasy Premier League data and generates optimized team recommendations using machine learning and optimization algorithms.

### 📋 Prerequisites
- Windows 10/11
- Internet connection
- Fantasy Premier League account

### 🛠️ First Time Setup

1. **Download and extract** the FPL Weekly Updater files to any folder
2. **Run the setup script**:
   ```cmd
   python setup_fpl_updater.py
   ```
3. **Follow the prompts** to:
   - Enter your FPL email and team ID
   - Set your FPL password (stored securely)
   - Configure your Perplexity API key (optional, for enhanced news analysis)
   - Choose where to save reports

### 🎯 Using the Application

#### Generate Weekly Report
```cmd
.\dist\FPLWeeklyUpdater.exe
```

#### Generate Appendix Only (ML Pipeline Report)
```cmd
.\dist\FPLWeeklyUpdater.exe appendix
```

#### Run Backtest Analysis
```cmd
.\dist\FPLWeeklyUpdater.exe backtest
```

### 📊 What You Get

**Weekly Report** (`fpl_weekly_update_YYYYMMDD_HHMMSS.pdf`):
- Optimal team selection for the next gameweek
- Player analysis with injury news and form
- Transfer recommendations
- Expected points predictions

**Appendix Report** (`backtest_analysis_lite_YYYYMMDD_HHMMSS.pdf`):
- Detailed model performance metrics
- Feature importance analysis
- Historical prediction accuracy

### 🔧 Configuration

The application reads configuration from:
- **`.env` file** - API keys and settings
- **Windows Credential Manager** - FPL password (secure storage)
- **`config.yaml`** - Application settings

### 🔐 Security

- ✅ FPL passwords stored securely in Windows Credential Manager
- ✅ API keys stored in local .env file (not shared)
- ✅ No data sent to external servers except configured APIs
- ✅ All processing happens locally on your machine

### 🆘 Troubleshooting

**"No password found" error:**
- Run `python setup_fpl_updater.py` again
- Make sure you're using the same email address

**"API key not found" error:**
- Edit the `.env` file manually
- Add: `PERPLEXITY_API_KEY=your_key_here`

**"Browser login failed":**
- Check your FPL credentials in Windows Credential Manager
- Try using Chrome instead of Edge (edit .env: `FPL_BROWSER=chrome`)

**Reports not generating:**
- Check that your FPL team ID is correct
- Make sure you're logged into FPL in your browser

### 📞 Support

For issues:
1. Check the log file: `fpl_weekly_update.log`
2. Run setup again: `python setup_fpl_updater.py`
3. Verify your credentials in Windows Credential Manager

### 🔄 Updates

To update the application:
1. Download the new version
2. Replace the old files
3. Run `python setup_fpl_updater.py` again if needed
4. Your settings will be preserved

---

**Happy optimizing! 🏆**
