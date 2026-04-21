from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

from src.services.auth_services import create_access_token, authenticate_user
from src.models.auth_models import Token



router_auth = APIRouter(prefix="/auth")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

@router_auth.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):

    """
    Método para validação de usuário e senha, e geração de token
    """

    try:
        user = authenticate_user(form_data.username, form_data.password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token(username=user["username"])

        return Token(access_token=access_token, token_type="bearer")

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")