from typing import Optional
from pydantic import BaseModel, Field

class EmployeeProfile(BaseModel):
    """Data schema representing a verified individual employee profile."""
    name: str = Field(description="Full name of the verified individual person")
    role: str = Field(default="", description="Role or professional title (e.g. CEO, VP of Engineering)")
    company: str = Field(description="Company association confirmed from source")
    seniority: Optional[str] = Field(default="Senior Leadership", description="Seniority tier: C-Suite, Founder, VP/Director, Lead/Manager")
    source_url: str = Field(description="URL where profile was found")
    source_domain: Optional[str] = Field(default="", description="Platform or domain where profile was located")
    source_snippet: Optional[str] = Field(default="", description="Snippet supporting the employment details")


class AgentState(BaseModel):
    """Schema for tracking agent execution state and user queries."""
    company_or_url: str
    role_preference: Optional[str] = None
