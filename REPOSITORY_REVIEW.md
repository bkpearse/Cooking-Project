# Repository Review: Project Cook

**Review Date:** December 7, 2025
**Reviewer:** Claude Code
**Project:** Cooking Recipe Classifier & Search - Le Wagon Data Science Bootcamp Final Project

---

## Executive Summary

Project Cook is a multi-feature cooking application that combines image classification for dish recognition with recipe search functionality. The application has been recently refactored to support Streamlit deployment with improved error handling and optional dependency management. While the core functionality is solid, there are several areas for improvement in code quality, security, and maintainability.

**Overall Assessment:** ⭐⭐⭐ (3/5)

---

## Strengths

### 1. **Well-Structured Refactoring**
- Recent Streamlit deployment preparation (commit 339c240) shows thoughtful architectural improvements
- Lazy loading of heavy dependencies (TensorFlow, Google Translator)
- Clear separation of concerns with `project_cook/interface/main.py` serving as a clean API layer
- Graceful degradation when optional assets are missing

### 2. **Good Documentation**
- Clear README with deployment instructions
- Inline documentation in `streamlit_app.py` and `interface/main.py`
- Configuration via environment variables is well-documented

### 3. **User Experience**
- Helpful error messages guide users to configure missing assets
- Multi-language support (6 languages)
- Dual functionality: image classification and recipe search

---

## Critical Issues

### 🔴 Security Concerns

#### 1. **Exposed Credentials in Repository**
**Severity: CRITICAL**

Multiple environment files are tracked in git:
- `.env` (line 190 in requirements.txt)
- `.env_vincent` contains GCP project IDs and bucket names
- These files should be in `.gitignore` (they are, but already committed)

**Impact:** Exposed GCP project credentials could lead to unauthorized access.

**Recommendation:**
```bash
# Remove from git history
git rm --cached .env .env_vincent
git commit -m "Remove sensitive environment files"

# Ensure .gitignore is properly configured (it is on line 5 and 38)
# Inform team members to rotate any exposed credentials
```

#### 2. **Wildcard Imports**
**Severity: MEDIUM**

Found in multiple files:
- `project_cook/utils.py:4` - `from project_cook.params import *`
- `project_cook/logic/registry.py:9` - `from project_cook.params import *`
- `project_cook/logic/data.py:8,13,15` - Multiple wildcard imports

**Impact:**
- Namespace pollution
- Difficult to track dependencies
- Potential name conflicts
- Security risk if malicious code is injected into imported modules

**Recommendation:** Replace with explicit imports:
```python
# Bad
from project_cook.params import *

# Good
from project_cook.params import CHUNK_SIZE, BUCKET_NAME, GCP_PROJECT
```

#### 3. **Broad Exception Handling**
In `project_cook/interface/main.py:323`:
```python
except Exception:  # pragma: no cover
    return text
```

**Impact:** Silently swallows all errors, making debugging difficult.

**Recommendation:** Catch specific exceptions or at least log the error.

---

## Code Quality Issues

### 1. **Duplicate Imports**
In `project_cook/logic/data.py`:
- Line 8: `from clean_text import *`
- Line 15: `from clean_text import *` (duplicate)

### 2. **Inconsistent Import Patterns**
- Some files use relative imports, others use absolute
- `data.py:8` imports `clean_text` without package prefix
- Should standardize on absolute imports from package root

### 3. **Missing Type Hints**
- Most functions lack type hints
- Recent code in `interface/main.py` has good type hints, but legacy code doesn't
- Consider gradually adding type hints for better maintainability

### 4. **Deprecated TensorFlow Import**
In `project_cook/logic/model.py:11-12`:
```python
from keras import layers, models, optimizers
from keras.models import Sequential
```

**Issue:** Using standalone Keras instead of `tensorflow.keras`

**Recommendation:** Update to:
```python
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
```

### 5. **Missing Tests**
- No test files found in the repository
- `.gitignore:42` actually excludes `tests/` directory
- Critical for production deployment

---

## Dependency Issues

### 1. **Outdated Dependencies**
- `tensorflow==2.10.0` (pinned, released 2022)
- Python 3.10.6 (specified in Dockerfile)
- `wrapt==1.15.0` and `zipp==3.15.0` are pinned

**Recommendation:**
- Update to `tensorflow>=2.15.0` for security patches
- Use dependency scanning tools (Dependabot, Safety)

### 2. **Duplicate Dependencies**
- `scikit-learn` appears twice in requirements.txt (lines 23 and 32)

### 3. **Problematic Dependency**
- `googletrans==4.0.0-rc1` is a release candidate and known to be unstable
- Consider using official Google Cloud Translation API instead

### 4. **Unnecessary Production Dependencies**
Development tools in production requirements:
- `jupyterlab`, `ipdb`, `ipykernel`, `yapf`, `pytest`, `nbresult`

**Recommendation:** Split into:
- `requirements.txt` (production)
- `requirements-dev.txt` (development)

---

## Architecture Issues

### 1. **Inconsistent Configuration Management**
Three different approaches to configuration:
- Environment variables (recommended, in `interface/main.py`)
- Direct file imports (`params.py`)
- Hardcoded values in legacy code

**Recommendation:** Consolidate to environment variables with sensible defaults.

### 2. **Unused Code**
In `project_cook/params.py`:
- Many variables are defined but `LOCAL_REGISTRY_PATH` is referenced in `registry.py` but never defined
- `BUCKET_NAME` is defined twice (line 14 and 19)

### 3. **Dead Code in Makefile**
Lines 12-24 are commented out, suggesting incomplete migration.

### 4. **Dockerfile Issues**

**Current Dockerfile issues:**
- Copies large directories (`new_index`, `notebooks/images`) into image
- No multi-stage build to reduce image size
- Running as root (security concern)
- Hard-coded paths

**Recommendation:**
```dockerfile
# Use multi-stage build
FROM python:3.10.6-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10.6-slim
# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY project_cook ./project_cook
COPY setup.py .
RUN chown -R appuser:appuser /app
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
CMD ["uvicorn", "project_cook.api_folder.api_file:api", "--host", "0.0.0.0", "--port", "8000"]
```

---

## API Issues

### 1. **Missing Input Validation**
In `project_cook/api_folder/api_file.py:14-16`:
```python
@api.get("/query-recipes/{lang}")
async def what_to_eat(lang: str, ingredients):
    return search_recipes(ingredients, lang)
```

**Issues:**
- No type hint for `ingredients`
- No validation of `lang` parameter (should be in SUPPORTED_LANGUAGES)
- No error handling
- Missing request body definition

### 2. **Duplicate Function Names**
Lines 15 and 19 both define `what_to_eat()` - second one shadows the first.

### 3. **Missing API Documentation**
- No OpenAPI/Swagger descriptions
- No request/response models using Pydantic

**Recommendation:**
```python
from typing import List
from pydantic import BaseModel
from fastapi import HTTPException, Query

class PredictionResponse(BaseModel):
    prediction: List[str]
    translation: List[str]

@api.post("/what-to-eat/{lang}", response_model=PredictionResponse)
async def predict_dish(
    lang: str = Path(..., regex="^(en|es|fr|de|it|pt)$"),
    img: UploadFile = File(...)
):
    """Classify a dish image and return predicted recipe names."""
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    img_bytes = await img.read()
    return pred_streamlit(img_bytes, lang)
```

---

## Best Practices Violations

### 1. **Notebook Checkpoints in Git**
- `.ipynb_checkpoints/` directories are tracked
- Should be in `.gitignore` (they are, but were committed before)

### 2. **Large Binary Files**
- `notebooks/model_weights.zip` (97MB) should use Git LFS
- Not ideal for git history

### 3. **Commented Code**
- Makefile has many commented lines
- Should be removed or properly documented

### 4. **Inconsistent Code Style**
- Mix of single and double quotes
- Inconsistent spacing
- No evidence of linting enforcement

**Recommendation:** Set up pre-commit hooks:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

---

## Missing Features

### 1. **Error Logging**
- No structured logging
- Difficult to debug production issues

### 2. **Monitoring & Metrics**
- No health check endpoint
- No metrics collection (Prometheus, etc.)

### 3. **CI/CD**
- No GitHub Actions or CI/CD pipeline
- Manual testing required

### 4. **Rate Limiting**
- API has no rate limiting
- Vulnerable to abuse

---

## Positive Recent Changes

The recent commit (339c240) "Prepare project for Streamlit deployment" shows excellent improvements:

1. ✅ Lazy dependency loading
2. ✅ Environment-based configuration
3. ✅ Clear error messages
4. ✅ Optional asset handling
5. ✅ Type hints in new code
6. ✅ Comprehensive docstrings

This suggests the team is moving in the right direction.

---

## Recommendations by Priority

### 🔴 **Critical (Do Immediately)**

1. **Remove credentials from git**
   - Remove `.env` and `.env_vincent` from repository
   - Rotate any exposed GCP credentials
   - Add template files (`.env.example`) instead

2. **Fix security vulnerabilities**
   - Replace wildcard imports with explicit imports
   - Update TensorFlow to latest version
   - Add input validation to API endpoints

3. **Fix duplicate function names in API**
   - Rename or consolidate the two `what_to_eat()` functions

### 🟡 **Important (Do This Week)**

4. **Add basic tests**
   - Start with integration tests for main workflows
   - Test API endpoints
   - Test image prediction and recipe search

5. **Split dependencies**
   - Create `requirements-dev.txt`
   - Remove dev dependencies from production

6. **Improve Dockerfile**
   - Use multi-stage builds
   - Run as non-root user
   - Reduce image size

7. **Add CI/CD pipeline**
   - GitHub Actions for testing
   - Automated linting
   - Security scanning

### 🟢 **Nice to Have (Do This Month)**

8. **Add logging and monitoring**
   - Structured logging (JSON format)
   - Health check endpoints
   - Metrics collection

9. **Improve API documentation**
   - Add Pydantic models
   - OpenAPI descriptions
   - Example requests/responses

10. **Code quality improvements**
    - Set up pre-commit hooks
    - Add type hints to legacy code
    - Remove commented code

11. **Performance optimization**
    - Add caching for predictions
    - Optimize Docker image size
    - Consider async database queries

---

## Code Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Python LOC | 716 | ✅ Manageable |
| Test Coverage | 0% | 🔴 Critical |
| Documentation | Partial | 🟡 Needs work |
| Type Hints | ~30% | 🟡 Improving |
| Security Issues | 5+ | 🔴 Critical |
| Dependency Health | Outdated | 🟡 Needs update |

---

## Conclusion

Project Cook demonstrates solid ML engineering fundamentals and shows promising recent improvements toward production-ready deployment. The Streamlit integration is well-executed with thoughtful error handling and configuration management.

However, **critical security issues must be addressed immediately**, particularly the exposed credentials and wildcard imports. The lack of tests is a significant concern for production deployment.

The codebase shows evidence of being a learning project (Le Wagon bootcamp), with typical early-stage issues like inconsistent patterns and missing production safeguards. With focused effort on the critical recommendations, this could become a production-ready application.

**Recommended Next Steps:**
1. Address all critical security issues (this week)
2. Add basic test coverage (>50% by end of month)
3. Set up CI/CD pipeline
4. Implement monitoring and logging
5. Plan for incremental improvements to code quality

---

## Additional Resources

- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)
- [TensorFlow Model Deployment Guide](https://www.tensorflow.org/tfx/serving/docker)
- [Python Security Best Practices](https://owasp.org/www-project-top-10/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

---

**Review Status:** Complete
**Follow-up Review Recommended:** After addressing critical issues
