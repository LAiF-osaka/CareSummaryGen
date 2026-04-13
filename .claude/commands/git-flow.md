# Git Flow Branching Model

This command provides comprehensive guidance for implementing the Git Flow branching model in your repository.

## Branch Types and Purposes

### Main Branches (Permanent)

#### main (or master)

- **Purpose**: Production-ready code only
- **Lifetime**: Permanent
- **Protected**: Yes, require PR reviews
- **Tagging**: Tag all releases (e.g., v1.0.0, v2.1.0)
- **Merge from**: hotfix, release branches only

#### develop

- **Purpose**: Integration branch for features
- **Lifetime**: Permanent
- **Protected**: Yes, require PR reviews
- **Merges**: All completed features, bugfixes
- **Always**: Ahead of main (except after hotfix)

### Supporting Branches (Temporary)

#### feature/* branches

- **Purpose**: Develop new features
- **Branch from**: develop
- **Merge back to**: develop
- **Naming**: `feature/<short-description>`
- **Examples**:
  - `feature/user-authentication`
  - `feature/add-payment-gateway`
  - `feature/api-rate-limiting`

**Workflow**:

```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# Work on feature...
# Commit regularly with conventional commits

# Merge back to develop
git checkout develop
git pull origin develop
git merge --no-ff feature/new-feature
git push origin develop
git branch -d feature/new-feature
```

#### bugfix/* branches

- **Purpose**: Fix bugs in develop branch
- **Branch from**: develop
- **Merge back to**: develop
- **Naming**: `bugfix/<short-description>`
- **Examples**:
  - `bugfix/fix-login-validation`
  - `bugfix/correct-timezone-display`

**Workflow**: Same as feature branches

#### release/* branches

- **Purpose**: Prepare for production release
- **Branch from**: develop
- **Merge to**: main AND develop
- **Naming**: `release/<version>`
- **Examples**:
  - `release/v1.2.0`
  - `release/v2.0.0-beta.1`

**Workflow**:

```bash
# Create release branch (only version bumps, bug fixes allowed)
git checkout develop
git checkout -b release/v1.2.0

# Finalize version number, update CHANGELOG, bug fixes only
# Commit: "chore(release): bump version to 1.2.0"

# Merge to main
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.2.0
git push origin develop

# Delete release branch
git branch -d release/v1.2.0
```

#### hotfix/* branches

- **Purpose**: Emergency fixes for production
- **Branch from**: main
- **Merge to**: main AND develop
- **Naming**: `hotfix/<version>` or `hotfix/<description>`
- **Examples**:
  - `hotfix/v1.2.1`
  - `hotfix/security-patch-xss`

**Workflow**:

```bash
# Create hotfix branch
git checkout main
git checkout -b hotfix/v1.2.1

# Fix the critical bug
# Commit: "fix: patch critical security vulnerability"

# Merge to main
git checkout main
git merge --no-ff hotfix/v1.2.1
git tag -a v1.2.1 -m "Hotfix version 1.2.1"
git push origin main --tags

# Merge to develop (or release branch if exists)
git checkout develop
git merge --no-ff hotfix/v1.2.1
git push origin develop

# Delete hotfix branch
git branch -d hotfix/v1.2.1
```

## Branch Naming Conventions

### Format

`<type>/<descriptive-name>`

### Rules

- Use lowercase letters only
- Separate words with hyphens (-)
- Keep names short but descriptive (3-5 words)
- Use present tense verbs for features
- Use imperative mood for fixes

### Good Examples

- `feature/user-authentication`
- `feature/add-dark-mode`
- `bugfix/fix-memory-leak`
- `hotfix/patch-security-vulnerability`
- `release/v2.0.0`

### Bad Examples

- `my-new-feature` (no type prefix)
- `feature/UserAuthentication` (use kebab-case)
- `feature/add_dark_mode` (use hyphens, not underscores)
- `fix-stuff` (not descriptive enough)
- `feature/implements-the-entire-new-payment-processing-system-with-stripe` (too long)

## Integration with Conventional Commits

Combine Git Flow with Conventional Commits for maximum clarity:

### Feature Branch Commits

```bash
git commit -m "feat(auth): implement OAuth 2.0 login flow"
git commit -m "feat(auth): add JWT token generation"
git commit -m "test(auth): add unit tests for login handler"
git commit -m "docs(auth): update API documentation"
```

### Bugfix Branch Commits

```bash
git commit -m "fix(api): correct response format"
git commit -m "test(api): add regression test for response"
```

### Release Branch Commits

```bash
git commit -m "chore(release): bump version to 2.0.0"
git commit -m "docs: update CHANGELOG for v2.0.0"
git commit -m "fix(deps): update vulnerable dependencies"
```

### Hotfix Branch Commits

```bash
git commit -m "fix(security): patch XSS vulnerability in user input"
git commit -m "chore: bump version to 1.2.1"
```

## Pull Request Guidelines

### Title Format

Use Conventional Commit format:

- `feat(scope): Add new feature X`
- `fix(scope): Resolve issue with Y`
- `release: Version 2.0.0`
- `hotfix: Security patch v1.2.1`

### Description Template

```markdown
## Type

- [ ] Feature
- [ ] Bugfix
- [ ] Release
- [ ] Hotfix

## Description

Brief description of changes...

## Related Issues

Closes #123
Relates to #456

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project conventions
- [ ] Documentation updated
- [ ] CHANGELOG updated (for releases)
- [ ] Version bumped (for releases/hotfixes)
```

## Version Numbering (Semantic Versioning)

Follow [Semantic Versioning 2.0.0](https://semver.org/):

### Format: MAJOR.MINOR.PATCH

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Pre-release versions:

- `2.0.0-alpha.1`
- `2.0.0-beta.2`
- `2.0.0-rc.1` (release candidate)

## Decision Tree: Which Branch to Create?

```
New work needed
    |
    |- Is it a critical production bug?
    |   -> Create hotfix/* from main
    |
    |- Are we preparing a release?
    |   -> Create release/* from develop
    |
    |- Is it a bug in develop branch?
    |   -> Create bugfix/* from develop
    |
    |- Is it a new feature or enhancement?
        -> Create feature/* from develop
```

## Best Practices

1. **Never commit directly to main or develop** - Always use PRs for code review
2. **Keep branches short-lived** - Features: days to 2 weeks max; Bugfixes: hours to days; Releases: 1-2 days; Hotfixes: hours
3. **Merge strategy** - Use `--no-ff` (no fast-forward) to preserve branch history
4. **Delete merged branches** - Clean up feature/bugfix branches after merging
5. **Tag all releases** - Annotated tags with release notes following semantic versioning
6. **Sync regularly** - Pull from develop before starting work; rebase long-running feature branches
7. **Document decisions** - Update CHANGELOG for releases; write meaningful commit messages

## When to Use This Command

Use `/git-flow` when:
- Creating new branches
- Planning release processes
- Resolving merge conflicts
- Deciding which branch to base work on
- Setting up repository branching strategy

Follow these guidelines consistently for a clean, maintainable Git history!
