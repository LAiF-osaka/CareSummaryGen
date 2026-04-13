# Conventional Commits Specification

This command provides comprehensive guidance for writing commit messages following the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/) specification.

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Components

#### 1. Type (Required)
The type of change being made.

**Common Types**:
- `feat`: A new feature for the user
- `fix`: A bug fix for the user
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (formatting, whitespace, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes to build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

#### 2. Scope (Optional)
The scope provides additional contextual information about what part of the codebase is affected.

**Examples**:
- `feat(auth): add OAuth 2.0 login`
- `fix(api): resolve race condition in webhook handler`
- `docs(readme): update installation instructions`

**Common Scopes for this Project**:

**Backend**: `api`, `auth`, `db`, `langchain`, `chromadb`, `model`
**Infrastructure**: `docker`, `ci`, `deploy`

#### 3. Subject (Required)
A brief description of the change.

**Rules**:
- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter
- No period (.) at the end
- Keep it under 50 characters
- Be specific and clear

**Good Examples**:
- `add user authentication with JWT`
- `fix memory leak in event listener`
- `update API response format`
- `remove deprecated methods`

**Bad Examples**:
- `Added new feature` (not imperative)
- `Fixes bugs` (too vague)
- `Update` (what was updated?)
- `WIP` (work in progress is not a complete change)

#### 4. Body (Optional)
Detailed explanation of the change.

**When to include**:
- Complex changes requiring explanation
- Non-obvious implementation decisions
- Context for why the change was needed
- Important considerations for reviewers

**Format**:
- Wrap at 72 characters per line
- Separate from subject with blank line
- Use bullet points for lists
- Explain WHAT and WHY, not HOW

**Example**:
```
feat(api): add rate limiting to public endpoints

Implement rate limiting to prevent API abuse and ensure fair usage
across all clients. Uses a sliding window algorithm with Redis for
distributed tracking.

- Public endpoints: 100 requests/minute
- Authenticated endpoints: 1000 requests/minute
- Admin endpoints: No limit

This addresses the recent spike in automated requests that caused
service degradation.
```

#### 5. Footer (Optional)
Additional metadata about the commit.

**Common Uses**:

**Breaking Changes**:
```
BREAKING CHANGE: Authentication now requires API key in header instead of query parameter.
```

**Issue References**:
```
Closes #123
Fixes #456, #789
Relates to #101
Refs #202
```

**Co-authors**:
```
Co-authored-by: Name <email@example.com>
```

## Breaking Changes

Breaking changes MUST be indicated in two ways:

### 1. Type/Scope with `!`
```
feat(api)!: change authentication method
```

### 2. Footer with BREAKING CHANGE
```
feat(api): change authentication method

BREAKING CHANGE: Authentication now requires API key in header.
All clients must update to include the `X-API-Key` header.
```

**Both can be used together** for maximum visibility:
```
feat(api)!: change authentication method

Migrate from query parameter to header-based authentication for
improved security and compliance with API best practices.

BREAKING CHANGE: The `api_key` query parameter is no longer supported.
All requests must include the `X-API-Key` header instead.

Migration guide: https://docs.example.com/migration/v2
```

## Complete Examples

### Simple Feature
```
feat(auth): add password reset functionality
```

### Bug Fix with Context
```
fix(api): prevent race condition in order processing

Add mutex lock around order creation to prevent duplicate orders
when multiple requests arrive simultaneously.

Closes #234
```

### Breaking Change
```
refactor(api)!: restructure user endpoints

BREAKING CHANGE: User endpoints moved from /api/users to /api/v2/users.
Update all API calls to use the new base path.

Closes #567
```

### Documentation Update
```
docs(readme): add troubleshooting section

Include common issues and solutions:
- Connection timeouts
- Authentication errors
- Rate limit handling
```

### Performance Improvement
```
perf(db): optimize user query with indexes

Add composite index on (email, active) columns to speed up
login queries by 85%.

Before: ~500ms average
After: ~75ms average

Refs #890
```

### Multiple Scopes
When a change affects multiple scopes, use the most prominent one or use a general scope like `core` or `app`:

```
refactor(core): restructure project architecture

Reorganize modules for better separation of concerns:
- Move shared utilities to /lib
- Extract API client to dedicated package
- Consolidate configuration files
```

## Type Selection Guide

### Decision Tree

```
What kind of change is it?

|- User-facing new feature?        -> feat
|- User-facing bug fix?            -> fix
|- Only documentation changed?     -> docs
|- Code formatting/style only?     -> style
|- Code restructuring (no change)? -> refactor
|- Performance improvement?        -> perf
|- Test-related changes?           -> test
|- Build system or dependencies?   -> build
|- CI/CD configuration?            -> ci
|- Other (tools, scripts, etc.)?   -> chore
|- Reverting previous commit?      -> revert
```

## Integration with Automated Tools

### Semantic Versioning
Automatically determine next version:
- `fix` -> PATCH (1.0.0 -> 1.0.1)
- `feat` -> MINOR (1.0.0 -> 1.1.0)
- `BREAKING CHANGE` -> MAJOR (1.0.0 -> 2.0.0)

## When to Use This Command

Use `/conventional-commits` when:
- Writing commit messages
- Reviewing pull requests
- Setting up commit hooks or linters
- Generating changelogs
- Determining semantic version bumps

Follow these guidelines for clear, consistent, and meaningful commit history!
