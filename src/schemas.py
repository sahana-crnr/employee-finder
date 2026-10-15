from typing import Optional
from pydantic import BaseModel, Field

class EmployeeProfile(BaseModel):
    """Data schema representing a verified individual employee profile."""
    name: str = Field(description="Full name of the verified individual person")
    job_title: str = Field(default="", description="Role or professional title (e.g. Senior Software Engineer)")
    company: str = Field(description="Company association confirmed from source")
    source_url: str = Field(description="URL where profile was found")
    source_snippet: str = Field(default="", description="Snippet supporting the employment details")


class AgentState(BaseModel):
    """Schema for tracking agent execution state and user queries."""
    company: str
    role_preference: Optional[str] = None

