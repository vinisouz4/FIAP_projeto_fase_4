from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List

from src.services.database_services import DatabaseService
from src.services.auth_services import current_user


router_update = APIRouter(prefix="/update")

user_dependency = Depends(current_user)

db_service = DatabaseService()


@router_update.get("/v1/update_bronze")
async def update_bronze(user: str = user_dependency):
    """
    Update the bronze table in the database.
    """
    try:
        db_service.update_bronze_table()
        return {"message": "Bronze table updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
