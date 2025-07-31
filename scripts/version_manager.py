#!/usr/bin/env python3
"""
Script Version Manager for Foundry
Handles script versioning, migration, and archival
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptVersionManager:
    """Manages script versions, migrations, and archival"""
    
    def __init__(self, scripts_dir: Path = None):
        self.scripts_dir = scripts_dir or Path(__file__).parent
        self.active_dir = self.scripts_dir / "active"
        self.archive_dir = self.scripts_dir / "archive"
        self.experimental_dir = self.scripts_dir / "experimental"
        self.migration_log_path = self.archive_dir / "migration_log.json"
        
        # Create directories if they don't exist
        for dir_path in [self.active_dir, self.archive_dir, self.experimental_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.migration_log = self._load_migration_log()
        
    def _load_migration_log(self) -> Dict:
        """Load migration history"""
        if self.migration_log_path.exists():
            with open(self.migration_log_path, 'r') as f:
                return json.load(f)
        return {"migrations": []}
    
    def _save_migration_log(self):
        """Save migration history"""
        with open(self.migration_log_path, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_script_metadata(self, script_path: Path) -> Dict:
        """Extract metadata from script"""
        metadata = {
            "path": str(script_path),
            "size": script_path.stat().st_size,
            "modified": datetime.fromtimestamp(script_path.stat().st_mtime).isoformat(),
            "checksum": self._calculate_checksum(script_path)
        }
        
        # Extract docstring and version info
        try:
            with open(script_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Look for version in comments
                for line in lines[:20]:  # Check first 20 lines
                    if 'version' in line.lower() and '=' in line:
                        version_parts = line.split('=')
                        if len(version_parts) > 1:
                            metadata['version'] = version_parts[1].strip().strip('"\'')
                            break
                
                # Extract docstring
                if '"""' in content:
                    start = content.find('"""') + 3
                    end = content.find('"""', start)
                    if end > start:
                        metadata['description'] = content[start:end].strip()
                        
        except Exception as e:
            logger.warning(f"Could not extract metadata from {script_path}: {e}")
            
        return metadata
    
    def identify_script_versions(self, base_name: str) -> List[Path]:
        """Find all versions of a script"""
        versions = []
        
        # Search patterns
        patterns = [
            f"{base_name}.py",
            f"{base_name}_*.py",
            f"*_{base_name}.py"
        ]
        
        for pattern in patterns:
            versions.extend(self.scripts_dir.glob(pattern))
            
        return sorted(set(versions))
    
    def consolidate_script(self, base_name: str, primary_version: str) -> Dict:
        """Consolidate script versions"""
        result = {
            "base_name": base_name,
            "primary_version": primary_version,
            "archived": [],
            "errors": []
        }
        
        try:
            # Find all versions
            versions = self.identify_script_versions(base_name)
            
            # Identify primary script
            primary_path = self.scripts_dir / primary_version
            if not primary_path.exists():
                result["errors"].append(f"Primary version {primary_version} not found")
                return result
            
            # Create archive directory for this month
            archive_month = self.archive_dir / datetime.now().strftime("%Y-%m")
            archive_month.mkdir(exist_ok=True)
            
            # Move primary to active directory
            active_path = self.active_dir / f"{base_name}.py"
            shutil.copy2(primary_path, active_path)
            result["active_path"] = str(active_path)
            
            # Archive other versions
            for version_path in versions:
                if version_path.name != primary_version:
                    archive_path = archive_month / version_path.name
                    shutil.move(str(version_path), str(archive_path))
                    
                    # Record in migration log
                    self.migration_log["migrations"].append({
                        "timestamp": datetime.now().isoformat(),
                        "action": "archive",
                        "script": version_path.name,
                        "from": str(version_path),
                        "to": str(archive_path),
                        "metadata": self._get_script_metadata(archive_path)
                    })
                    
                    result["archived"].append(str(archive_path))
            
            # Save migration log
            self._save_migration_log()
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Error consolidating {base_name}: {e}")
            
        return result
    
    def create_compatibility_wrapper(self, old_script: str, new_script: str) -> str:
        """Create a compatibility wrapper for old script names"""
        wrapper_content = f'''#!/usr/bin/env python3
"""
Compatibility wrapper for {old_script}
This script has been consolidated to {new_script}
"""

import sys
import os
from pathlib import Path

# Add active directory to path
active_dir = Path(__file__).parent / "active"
sys.path.insert(0, str(active_dir))

# Import from new location
from {Path(new_script).stem} import *

if __name__ == "__main__":
    import warnings
    warnings.warn(
        f"{{old_script}} is deprecated. Please use {{new_script}} directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Forward to main if it exists
    if 'main' in globals():
        main()
'''
        return wrapper_content
    
    def generate_migration_report(self) -> str:
        """Generate a migration status report"""
        report = []
        report.append("# Script Version Migration Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary statistics
        total_migrations = len(self.migration_log.get("migrations", []))
        report.append(f"Total migrations: {total_migrations}")
        report.append("")
        
        # Active scripts
        report.append("## Active Scripts")
        if self.active_dir.exists():
            for script in sorted(self.active_dir.glob("*.py")):
                metadata = self._get_script_metadata(script)
                report.append(f"- **{script.name}**")
                report.append(f"  - Size: {metadata['size']} bytes")
                report.append(f"  - Modified: {metadata['modified']}")
                if 'version' in metadata:
                    report.append(f"  - Version: {metadata['version']}")
                if 'description' in metadata:
                    report.append(f"  - Description: {metadata['description'][:100]}...")
        report.append("")
        
        # Recent migrations
        report.append("## Recent Migrations")
        recent = sorted(
            self.migration_log.get("migrations", []),
            key=lambda x: x["timestamp"],
            reverse=True
        )[:10]
        
        for migration in recent:
            report.append(f"- {migration['timestamp']}: {migration['action']} {migration['script']}")
            
        return "\n".join(report)
    
    def validate_consolidation(self) -> List[Dict]:
        """Validate the consolidation process"""
        issues = []
        
        # Check for duplicate functionality
        active_scripts = list(self.active_dir.glob("*.py"))
        for i, script1 in enumerate(active_scripts):
            for script2 in active_scripts[i+1:]:
                # Compare checksums
                if self._calculate_checksum(script1) == self._calculate_checksum(script2):
                    issues.append({
                        "type": "duplicate",
                        "severity": "warning",
                        "scripts": [str(script1), str(script2)],
                        "message": "Scripts have identical content"
                    })
        
        # Check for missing dependencies
        for script in active_scripts:
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    
                # Check imports
                imports = [line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
                for imp in imports:
                    if '../' in imp or 'scripts.' in imp:
                        issues.append({
                            "type": "import_path",
                            "severity": "error",
                            "script": str(script),
                            "line": imp,
                            "message": "Script uses relative imports that may break after consolidation"
                        })
            except Exception as e:
                issues.append({
                    "type": "validation_error",
                    "severity": "error",
                    "script": str(script),
                    "message": str(e)
                })
                
        return issues


# Main consolidation execution
if __name__ == "__main__":
    manager = ScriptVersionManager()
    
    # Define primary versions based on analysis
    consolidation_plan = {
        'teacher_pair_generation': 'teacher_pair_generation_vllm_ssh.py',
        'data_enrichment': 'data_enrichment_enhanced_gpu_fixed_v2.py',
        'distillation': 'distillation_vllm_faster_improved.py',
        'student_self_study': 'student_self_study_enhanced.py'
    }
    
    print("Starting script consolidation...")
    
    for base_name, primary_version in consolidation_plan.items():
        result = manager.consolidate_script(base_name, primary_version)
        print(f"\nConsolidated {base_name}:")
        print(f"  Primary: {result.get('active_path', 'N/A')}")
        print(f"  Archived: {len(result['archived'])} versions")
        if result['errors']:
            print(f"  Errors: {', '.join(result['errors'])}")
    
    # Generate report
    report = manager.generate_migration_report()
    report_path = manager.scripts_dir / "migration_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nMigration report saved to: {report_path}")
    
    # Validate
    issues = manager.validate_consolidation()
    if issues:
        print("\nValidation Issues Found:")
        for issue in issues:
            print(f"  [{issue['severity']}] {issue['type']}: {issue['message']}")