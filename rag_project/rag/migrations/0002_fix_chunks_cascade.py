from django.db import migrations


def update_chunks_fk_cascade(apps, schema_editor):
    if schema_editor.connection.vendor == 'postgresql':
        with schema_editor.connection.cursor() as cursor:
            cursor.execute("""
                DO $$
                DECLARE
                    r RECORD;
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks') THEN
                        FOR r IN (
                            SELECT tc.constraint_name
                            FROM information_schema.table_constraints AS tc
                            JOIN information_schema.key_column_usage AS kcu
                              ON tc.constraint_name = kcu.constraint_name
                              AND tc.table_schema = kcu.table_schema
                            JOIN information_schema.constraint_column_usage AS ccu
                              ON ccu.constraint_name = tc.constraint_name
                              AND ccu.table_schema = tc.table_schema
                            WHERE tc.constraint_type = 'FOREIGN KEY'
                              AND tc.table_name = 'chunks'
                              AND ccu.table_name = 'documents'
                        ) LOOP
                            EXECUTE 'ALTER TABLE chunks DROP CONSTRAINT ' || quote_ident(r.constraint_name);
                        END LOOP;

                        -- Purge any orphaned chunks before creating FK constraint
                        DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents);

                        ALTER TABLE chunks ADD CONSTRAINT chunks_document_id_421b6371_fk_documents_id
                            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE;
                    END IF;
                END $$;
            """)


def reverse_chunks_fk_cascade(apps, schema_editor):
    if schema_editor.connection.vendor == 'postgresql':
        with schema_editor.connection.cursor() as cursor:
            cursor.execute("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chunks') THEN
                        ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_document_id_421b6371_fk_documents_id;
                        ALTER TABLE chunks ADD CONSTRAINT chunks_document_id_421b6371_fk_documents_id
                            FOREIGN KEY (document_id) REFERENCES documents(id);
                    END IF;
                END $$;
            """)


class Migration(migrations.Migration):

    dependencies = [
        ('rag', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(update_chunks_fk_cascade, reverse_chunks_fk_cascade),
    ]
