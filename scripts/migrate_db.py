#!/usr/bin/env python3
"""
Database Migration Script for Mark1 Agent System

This script handles database schema creation, updates, and migrations.
It provides a command-line interface for database management operations.

Usage:
    python scripts/migrate_db.py init          # Initialize database
    python scripts/migrate_db.py migrate       # Run pending migrations
    python scripts/migrate_db.py reset         # Reset database (DANGER)
    python scripts/migrate_db.py status        # Check migration status
    python scripts/migrate_db.py create <name> # Create new migration
"""

import os
import sys
import argparse
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mark1.config.settings import get_settings
from mark1.config.logging_config import setup_logging
from mark1.storage.database import DatabaseManager
from mark1.utils.exceptions import DatabaseError, MigrationError


class MigrationManager:
    """Manages database migrations for the Mark1 system."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
        
    def init_migration_table(self):
        """Initialize the migrations tracking table."""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS migrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        checksum TEXT
                    )
                """)
                conn.commit()
                self.logger.info("Migration tracking table initialized")
        except Exception as e:
            raise MigrationError(f"Failed to initialize migration table: {e}")
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM migrations ORDER BY applied_at")
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Migration table doesn't exist yet
            return []
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        applied = set(self.get_applied_migrations())
        available = set(self._get_available_migrations())
        return sorted(list(available - applied))
    
    def _get_available_migrations(self) -> List[str]:
        """Get list of all available migration files."""
        migrations = []
        for file_path in self.migrations_dir.glob("*.sql"):
            if file_path.stem != "__init__":
                migrations.append(file_path.stem)
        return sorted(migrations)
    
    def apply_migration(self, migration_name: str) -> bool:
        """Apply a single migration."""
        migration_file = self.migrations_dir / f"{migration_name}.sql"
        
        if not migration_file.exists():
            raise MigrationError(f"Migration file not found: {migration_file}")
        
        try:
            # Read migration SQL
            sql_content = migration_file.read_text()
            checksum = str(hash(sql_content))
            
            with self.db_manager.get_connection() as conn:
                # Execute migration SQL
                conn.executescript(sql_content)
                
                # Record migration as applied
                conn.execute(
                    "INSERT INTO migrations (name, checksum) VALUES (?, ?)",
                    (migration_name, checksum)
                )
                conn.commit()
                
            self.logger.info(f"Applied migration: {migration_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply migration {migration_name}: {e}")
            raise MigrationError(f"Migration failed: {e}")
    
    def migrate(self) -> int:
        """Apply all pending migrations."""
        self.init_migration_table()
        pending = self.get_pending_migrations()
        
        if not pending:
            self.logger.info("No pending migrations")
            return 0
        
        applied_count = 0
        for migration in pending:
            try:
                self.apply_migration(migration)
                applied_count += 1
            except MigrationError:
                self.logger.error(f"Migration stopped at: {migration}")
                break
        
        self.logger.info(f"Applied {applied_count} migrations")
        return applied_count
    
    def create_migration(self, name: str) -> Path:
        """Create a new migration file."""
        # Generate timestamp prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{name}.sql"
        migration_file = self.migrations_dir / filename
        
        # Create migration template
        template = f"""-- Migration: {name}
-- Created: {datetime.now().isoformat()}
-- Description: Add your migration description here

-- Add your SQL statements below
-- Example:
-- CREATE TABLE example (
--     id INTEGER PRIMARY KEY,
--     name TEXT NOT NULL
-- );

-- Remember to test your migration before applying!
"""
        
        migration_file.write_text(template)
        self.logger.info(f"Created migration file: {migration_file}")
        return migration_file
    
    def get_status(self) -> Dict[str, List[str]]:
        """Get migration status."""
        try:
            self.init_migration_table()
            applied = self.get_applied_migrations()
            pending = self.get_pending_migrations()
            
            return {
                "applied": applied,
                "pending": pending,
                "total": len(applied) + len(pending)
            }
        except Exception as e:
            raise MigrationError(f"Failed to get migration status: {e}")


class SchemaInitializer:
    """Handles initial database schema creation."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def create_initial_schema(self):
        """Create the initial database schema."""
        schema_sql = """
        -- Agents table
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT DEFAULT 'inactive',
            configuration TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP
        );

        -- Tasks table
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 0,
            data TEXT,
            result TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (agent_id) REFERENCES agents (id)
        );

        -- Context table
        CREATE TABLE IF NOT EXISTS contexts (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            session_id TEXT,
            type TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_id) REFERENCES agents (id)
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_agents_status ON agents (status);
        CREATE INDEX IF NOT EXISTS idx_agents_type ON agents (type);
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks (agent_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
        CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at);
        CREATE INDEX IF NOT EXISTS idx_contexts_agent_id ON contexts (agent_id);
        CREATE INDEX IF NOT EXISTS idx_contexts_session_id ON contexts (session_id);
        CREATE INDEX IF NOT EXISTS idx_contexts_key ON contexts (key);
        CREATE INDEX IF NOT EXISTS idx_contexts_expires_at ON contexts (expires_at);
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
                self.logger.info("Initial database schema created successfully")
        except Exception as e:
            raise DatabaseError(f"Failed to create initial schema: {e}")


def setup_database():
    """Setup database with initial schema and migration system."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)
    
    db_manager = DatabaseManager(settings.database_url)
    schema_init = SchemaInitializer(db_manager)
    migration_manager = MigrationManager(db_manager)
    
    return db_manager, schema_init, migration_manager


def cmd_init(args):
    """Initialize database with schema and migration system."""
    print("Initializing database...")
    
    try:
        db_manager, schema_init, migration_manager = setup_database()
        
        # Create initial schema
        schema_init.create_initial_schema()
        
        # Initialize migration system
        migration_manager.init_migration_table()
        
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


def cmd_migrate(args):
    """Run pending migrations."""
    print("Running database migrations...")
    
    try:
        _, _, migration_manager = setup_database()
        applied_count = migration_manager.migrate()
        
        if applied_count > 0:
            print(f"✅ Applied {applied_count} migrations")
        else:
            print("✅ No pending migrations")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


def cmd_status(args):
    """Show migration status."""
    try:
        _, _, migration_manager = setup_database()
        status = migration_manager.get_status()
        
        print(f"Migration Status:")
        print(f"  Applied: {len(status['applied'])}")
        print(f"  Pending: {len(status['pending'])}")
        print(f"  Total: {status['total']}")
        
        if status['pending']:
            print(f"\nPending migrations:")
            for migration in status['pending']:
                print(f"  - {migration}")
        
        if args.verbose and status['applied']:
            print(f"\nApplied migrations:")
            for migration in status['applied']:
                print(f"  ✅ {migration}")
                
    except Exception as e:
        print(f"❌ Failed to get status: {e}")
        sys.exit(1)


def cmd_create(args):
    """Create a new migration file."""
    if not args.name:
        print("❌ Migration name is required")
        sys.exit(1)
    
    try:
        _, _, migration_manager = setup_database()
        migration_file = migration_manager.create_migration(args.name)
        print(f"✅ Created migration: {migration_file}")
        print(f"📝 Edit the file to add your SQL statements")
        
    except Exception as e:
        print(f"❌ Failed to create migration: {e}")
        sys.exit(1)


def cmd_reset(args):
    """Reset database (DANGEROUS)."""
    if not args.force:
        response = input("⚠️  This will delete ALL data. Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            return
    
    try:
        settings = get_settings()
        db_path = Path(settings.database_url.replace('sqlite:///', ''))
        
        if db_path.exists():
            db_path.unlink()
            print("✅ Database file deleted")
        
        # Reinitialize
        cmd_init(args)
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mark1 Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize database')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Run pending migrations')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    status_parser.add_argument('-v', '--verbose', action='store_true', 
                              help='Show detailed status')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new migration')
    create_parser.add_argument('name', help='Migration name')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database (DANGEROUS)')
    reset_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Command dispatch
    commands = {
        'init': cmd_init,
        'migrate': cmd_migrate,  
        'status': cmd_status,
        'create': cmd_create,
        'reset': cmd_reset
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()