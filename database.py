import json
import pickle
import zlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
import streamlit as st
import hashlib

class DataManager:
    def __init__(self, data_dir: str = "saved_analyses", use_database: bool = True):
        self.data_dir = data_dir
        self.use_database = use_database
        os.makedirs(data_dir, exist_ok=True)
        
        # Configuración de la base de datos
        self.db_config = {
            'host': 'localhost',
            'database': 'whatsapp_osint',
            'user': 'admin',
            'password': 'ClaveSegura123',
            'charset': 'utf8mb4'
        }
        
        # Inicializar base de datos
        if use_database:
            self._initialize_database()
    
    def _get_connection(self):
        """Establece conexión con la base de datos"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Error as e:
            st.error(f"❌ Error de conexión a la base de datos: {e}")
            return None
    
    def _initialize_database(self):
        """Inicializa la base de datos con las tablas necesarias"""
        connection = self._get_connection()
        if not connection:
            st.warning("⚠️ Usando almacenamiento local (sin base de datos)")
            self.use_database = False
            return
        
        try:
            cursor = connection.cursor()
            
            # Tabla de análisis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    timestamp DATETIME NOT NULL,
                    total_messages INT,
                    total_users INT,
                    avg_sentiment FLOAT,
                    conversation_days INT,
                    tags JSON,
                    is_public BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de datos de análisis (datos comprimidos)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_data (
                    analysis_id VARCHAR(100) PRIMARY KEY,
                    compressed_data LONGBLOB NOT NULL,
                    data_size INT,
                    data_hash VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                )
            """)
            
            # Tabla de estadísticas de usuario
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    analysis_id VARCHAR(100),
                    user_name VARCHAR(255),
                    messages_count INT,
                    words_count INT,
                    avg_sentiment FLOAT,
                    communicator_type VARCHAR(100),
                    activity_level VARCHAR(50),
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                )
            """)
            
            # Índices para mejor performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_name ON analyses(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_stats_analysis ON user_stats(analysis_id)")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            st.success("✅ Base de datos inicializada correctamente")
            
        except Error as e:
            st.error(f"❌ Error inicializando base de datos: {e}")
            self.use_database = False
    
    def _compress_data(self, data: Dict[str, Any]) -> Tuple[bytes, int, str]:
        """Comprime los datos y calcula hash"""
        serialized_data = pickle.dumps(data)
        compressed_data = zlib.compress(serialized_data)
        data_hash = hashlib.sha256(serialized_data).hexdigest()
        return compressed_data, len(serialized_data), data_hash
    
    def _decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Descomprime los datos"""
        try:
            serialized_data = zlib.decompress(compressed_data)
            return pickle.loads(serialized_data)
        except Exception as e:
            raise Exception(f"Error descomprimiendo datos: {e}")
    
    def _extract_analysis_metadata(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae metadatos importantes del análisis"""
        metrics = analysis_data.get('metrics', {})
        user_profiles = analysis_data.get('user_profiles', {})
        messages = analysis_data.get('messages', [])
        
        # Calcular días de conversación
        conversation_days = 0
        if messages:
            try:
                timestamps = [msg.get('timestamp') for msg in messages if msg.get('timestamp')]
                if timestamps:
                    dates = [pd.to_datetime(ts).date() for ts in timestamps if pd.to_datetime(ts, errors='coerce') is not pd.NaT]
                    if dates:
                        conversation_days = len(set(dates))
            except:
                conversation_days = 0
        
        return {
            'total_messages': metrics.get('total_messages', 0),
            'total_users': metrics.get('total_users', 0),
            'avg_sentiment': metrics.get('avg_sentiment', 0),
            'conversation_days': conversation_days,
            'message_density': metrics.get('message_density', 0),
            'engagement_rate': metrics.get('engagement_rate', 0)
        }
    
    def save_analysis(self, name: str, analysis_data: Dict[str, Any], 
                     description: str = "", tags: List[str] = None, 
                     is_public: bool = False) -> str:
        """Guarda un análisis en la base de datos con metadatos enriquecidos"""
        # Generar ID único
        analysis_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Extraer metadatos
        metadata = self._extract_analysis_metadata(analysis_data)
        
        if self.use_database:
            return self._save_to_database(analysis_id, name, description, analysis_data, metadata, tags or [], is_public)
        else:
            return self._save_to_filesystem(analysis_id, name, analysis_data, metadata)
    
    def _save_to_database(self, analysis_id: str, name: str, description: str, 
                         analysis_data: Dict[str, Any], metadata: Dict[str, Any], 
                         tags: List[str], is_public: bool) -> str:
        """Guarda análisis en la base de datos MySQL"""
        connection = self._get_connection()
        if not connection:
            return self._save_to_filesystem(analysis_id, name, analysis_data, metadata)
        
        try:
            cursor = connection.cursor()
            
            # Comprimir datos
            compressed_data, original_size, data_hash = self._compress_data(analysis_data)
            
            # Insertar en tabla analyses
            cursor.execute("""
                INSERT INTO analyses (id, name, description, timestamp, total_messages, total_users, 
                                   avg_sentiment, conversation_days, tags, is_public)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                analysis_id, name, description, datetime.now().isoformat(),
                metadata['total_messages'], metadata['total_users'], metadata['avg_sentiment'],
                metadata['conversation_days'], json.dumps(tags), is_public
            ))
            
            # Insertar datos comprimidos
            cursor.execute("""
                INSERT INTO analysis_data (analysis_id, compressed_data, data_size, data_hash)
                VALUES (%s, %s, %s, %s)
            """, (analysis_id, compressed_data, original_size, data_hash))
            
            # Guardar estadísticas de usuarios
            user_profiles = analysis_data.get('user_profiles', {})
            for user_name, profile in user_profiles.items():
                cursor.execute("""
                    INSERT INTO user_stats (analysis_id, user_name, messages_count, words_count, 
                                         avg_sentiment, communicator_type, activity_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    analysis_id, user_name, profile.get('messages_count', 0),
                    profile.get('words_count', 0), profile.get('avg_sentiment', 0),
                    profile.get('communicator_type', ''), profile.get('activity_level', '')
                ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            st.success(f"✅ Análisis '{name}' guardado en base de datos (ID: {analysis_id})")
            return analysis_id
            
        except Error as e:
            st.error(f"❌ Error guardando en base de datos: {e}")
            connection.rollback()
            return self._save_to_filesystem(analysis_id, name, analysis_data, metadata)
    
    def _save_to_filesystem(self, analysis_id: str, name: str, 
                           analysis_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Guarda análisis en sistema de archivos (fallback)"""
        try:
            analysis_record = {
                'id': analysis_id,
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            # Guardar metadatos
            metadata_file = os.path.join(self.data_dir, f"{analysis_id}_meta.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_record, f, indent=2, ensure_ascii=False)
            
            # Guardar datos comprimidos
            compressed_data, _, _ = self._compress_data(analysis_data)
            data_file = os.path.join(self.data_dir, f"{analysis_id}_data.pkl.gz")
            with open(data_file, 'wb') as f:
                f.write(compressed_data)
            
            st.info(f"💾 Análisis '{name}' guardado localmente (ID: {analysis_id})")
            return analysis_id
            
        except Exception as e:
            st.error(f"❌ Error guardando en sistema de archivos: {e}")
            raise
    
    def load_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Carga un análisis desde la base de datos o sistema de archivos"""
        if self.use_database:
            result = self._load_from_database(analysis_id)
            if result:
                return result
        
        # Fallback a sistema de archivos
        return self._load_from_filesystem(analysis_id)
    
    def _load_from_database(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Carga análisis desde la base de datos MySQL"""
        connection = self._get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            # Obtener datos comprimidos
            cursor.execute("SELECT compressed_data FROM analysis_data WHERE analysis_id = %s", (analysis_id,))
            result = cursor.fetchone()
            
            if not result:
                return None
            
            compressed_data = result[0]
            analysis_data = self._decompress_data(compressed_data)
            
            cursor.close()
            connection.close()
            
            return analysis_data
            
        except Error as e:
            st.error(f"❌ Error cargando desde base de datos: {e}")
            return None
    
    def _load_from_filesystem(self, analysis_id: str) -> Dict[str, Any]:
        """Carga análisis desde sistema de archivos"""
        data_file = os.path.join(self.data_dir, f"{analysis_id}_data.pkl.gz")
        
        try:
            with open(data_file, 'rb') as f:
                compressed_data = f.read()
            return self._decompress_data(compressed_data)
        except FileNotFoundError:
            raise Exception(f"Análisis {analysis_id} no encontrado")
        except Exception as e:
            raise Exception(f"Error cargando análisis: {e}")
    
    def get_saved_analyses(self, search_query: str = "", tags: List[str] = None, 
                          date_range: Tuple[datetime, datetime] = None) -> Dict[str, Dict[str, Any]]:
        """Obtiene análisis guardados con filtros avanzados"""
        if self.use_database:
            return self._get_from_database(search_query, tags, date_range)
        else:
            return self._get_from_filesystem()
    
    def _get_from_database(self, search_query: str = "", tags: List[str] = None,
                          date_range: Tuple[datetime, datetime] = None) -> Dict[str, Dict[str, Any]]:
        """Obtiene análisis desde la base de datos con filtros"""
        connection = self._get_connection()
        if not connection:
            return self._get_from_filesystem()
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            query = """
                SELECT id, name, description, timestamp, total_messages, total_users, 
                       avg_sentiment, conversation_days, tags, is_public, created_at
                FROM analyses 
                WHERE 1=1
            """
            params = []
            
            # Filtro de búsqueda
            if search_query:
                query += " AND (name LIKE %s OR description LIKE %s)"
                params.extend([f"%{search_query}%", f"%{search_query}%"])
            
            # Filtro de tags
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("JSON_CONTAINS(tags, %s)")
                    params.append(json.dumps(tag))
                query += " AND (" + " OR ".join(tag_conditions) + ")"
            
            # Filtro de fecha
            if date_range:
                start_date, end_date = date_range
                query += " AND timestamp BETWEEN %s AND %s"
                params.extend([start_date.isoformat(), end_date.isoformat()])
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            analyses = {}
            
            for row in cursor.fetchall():
                analyses[row['id']] = {
                    'name': row['name'],
                    'description': row['description'],
                    'timestamp': row['timestamp'],
                    'metadata': {
                        'total_messages': row['total_messages'],
                        'total_users': row['total_users'],
                        'avg_sentiment': row['avg_sentiment'],
                        'conversation_days': row['conversation_days']
                    },
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'is_public': row['is_public'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None
                }
            
            cursor.close()
            connection.close()
            
            return analyses
            
        except Error as e:
            st.error(f"❌ Error obteniendo análisis desde base de datos: {e}")
            return self._get_from_filesystem()
    
    def _get_from_filesystem(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene análisis desde sistema de archivos"""
        analyses = {}
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_meta.json'):
                analysis_id = filename.replace('_meta.json', '')
                
                try:
                    with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        analyses[analysis_id] = metadata
                except Exception:
                    continue
        
        return dict(sorted(analyses.items(), key=lambda x: x[1]['timestamp'], reverse=True))
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Elimina un análisis de la base de datos o sistema de archivos"""
        if self.use_database:
            success = self._delete_from_database(analysis_id)
            if success:
                return True
        
        # Fallback a sistema de archivos
        return self._delete_from_filesystem(analysis_id)
    
    def _delete_from_database(self, analysis_id: str) -> bool:
        """Elimina análisis desde la base de datos MySQL"""
        connection = self._get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM analyses WHERE id = %s", (analysis_id,))
            connection.commit()
            cursor.close()
            connection.close()
            return True
            
        except Error as e:
            st.error(f"❌ Error eliminando desde base de datos: {e}")
            return False
    
    def _delete_from_filesystem(self, analysis_id: str) -> bool:
        """Elimina análisis desde sistema de archivos"""
        try:
            meta_file = os.path.join(self.data_dir, f"{analysis_id}_meta.json")
            data_file = os.path.join(self.data_dir, f"{analysis_id}_data.pkl.gz")
            
            if os.path.exists(meta_file):
                os.remove(meta_file)
            if os.path.exists(data_file):
                os.remove(data_file)
            
            return True
        except Exception:
            return False
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas avanzadas de la base de datos"""
        if self.use_database:
            return self._get_stats_from_database()
        else:
            return self._get_stats_from_filesystem()
    
    def _get_stats_from_database(self) -> Dict[str, Any]:
        """Obtiene estadísticas desde la base de datos MySQL"""
        connection = self._get_connection()
        if not connection:
            return self._get_stats_from_filesystem()
        
        try:
            cursor = connection.cursor()
            
            # Estadísticas generales
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_analyses,
                    SUM(total_messages) as total_messages,
                    AVG(avg_sentiment) as avg_sentiment_global,
                    MIN(timestamp) as oldest_analysis,
                    MAX(timestamp) as newest_analysis,
                    AVG(conversation_days) as avg_conversation_days
                FROM analyses
            """)
            stats_row = cursor.fetchone()
            
            # Usuarios más activos
            cursor.execute("""
                SELECT user_name, SUM(messages_count) as total_messages
                FROM user_stats 
                GROUP BY user_name 
                ORDER BY total_messages DESC 
                LIMIT 5
            """)
            top_users = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return {
                'total_analyses': stats_row[0] or 0,
                'total_messages': stats_row[1] or 0,
                'avg_sentiment_global': float(stats_row[2] or 0),
                'oldest_analysis': stats_row[3].isoformat() if stats_row[3] else None,
                'newest_analysis': stats_row[4].isoformat() if stats_row[4] else None,
                'avg_conversation_days': float(stats_row[5] or 0),
                'top_users': [{'user': user[0], 'messages': user[1]} for user in top_users],
                'storage_type': 'database'
            }
            
        except Error as e:
            st.error(f"❌ Error obteniendo estadísticas desde base de datos: {e}")
            return self._get_stats_from_filesystem()
    
    def _get_stats_from_filesystem(self) -> Dict[str, Any]:
        """Obtiene estadísticas desde sistema de archivos"""
        analyses = self._get_from_filesystem()
        
        total_messages = sum(meta['metadata']['total_messages'] for meta in analyses.values())
        total_sentiment = sum(meta['metadata']['avg_sentiment'] for meta in analyses.values())
        
        return {
            'total_analyses': len(analyses),
            'total_messages': total_messages,
            'avg_sentiment_global': total_sentiment / len(analyses) if analyses else 0,
            'oldest_analysis': min(meta['timestamp'] for meta in analyses.values()) if analyses else None,
            'newest_analysis': max(meta['timestamp'] for meta in analyses.values()) if analyses else None,
            'storage_type': 'filesystem'
        }
    
    def backup_database(self, backup_path: str) -> bool:
        """Crea un backup de la base de datos"""
        if not self.use_database:
            st.warning("⚠️ Backup solo disponible con base de datos")
            return False
        
        try:
            # Esto es un ejemplo básico - en producción usarías mysqldump
            analyses = self._get_from_database()
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'total_analyses': len(analyses),
                'analyses': analyses
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Backup creado en: {backup_path}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error creando backup: {e}")
            return False
    
    def get_database_health(self) -> Dict[str, Any]:
        """Verifica la salud de la base de datos"""
        if not self.use_database:
            return {'status': 'filesystem', 'healthy': True}
        
        connection = self._get_connection()
        if not connection:
            return {'status': 'disconnected', 'healthy': False}
        
        try:
            cursor = connection.cursor()
            
            # Verificar tablas
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            expected_tables = ['analyses', 'analysis_data', 'user_stats']
            
            # Verificar integridad de datos
            cursor.execute("SELECT COUNT(*) FROM analyses")
            analysis_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_data")
            data_count = cursor.fetchone()[0]
            
            cursor.close()
            connection.close()
            
            return {
                'status': 'connected',
                'healthy': all(table in tables for table in expected_tables),
                'tables_missing': [table for table in expected_tables if table not in tables],
                'analysis_count': analysis_count,
                'data_count': data_count,
                'integrity_ok': analysis_count == data_count
            }
            
        except Error as e:
            return {'status': 'error', 'healthy': False, 'error': str(e)}