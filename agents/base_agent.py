"""
Base Agent Class for Trading Desk Agents

Provides common functionality for all specialized agents in the trading desk.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AgentRole(Enum):
    """Roles for trading desk agents"""
    MARKET_ANALYST = "market_analyst"
    STRATEGIST = "strategist"
    RISK_MANAGER = "risk_manager"
    EXECUTION = "execution"
    HEAD_TRADER = "head_trader"
    COORDINATOR = "coordinator"


class MessageType(Enum):
    """Types of inter-agent messages"""
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    STRATEGY_PROPOSAL = "strategy_proposal"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_SIGNAL = "trade_signal"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    MARKET_UPDATE = "market_update"
    ALERT = "alert"
    CHAT = "chat"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: AgentRole
    receiver: AgentRole
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=urgent
    requires_response: bool = False
    conversation_id: str = None
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender.value,
            "receiver": self.receiver.value,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "conversation_id": self.conversation_id
        }


@dataclass
class AgentState:
    """Current state of an agent"""
    role: AgentRole
    status: str = "idle"  # idle, processing, waiting, error
    current_task: str = None
    last_activity: str = None
    messages_processed: int = 0
    insights_generated: int = 0


class BaseAgent(ABC):
    """
    Abstract base class for all trading desk agents.
    
    Each agent has:
    - A specific role and expertise
    - Ability to communicate with other agents
    - Access to OpenRouter LLM for reasoning
    - Memory of recent conversations
    """
    
    def __init__(self, role: AgentRole, name: str = None):
        self.role = role
        self.name = name or role.value.replace("_", " ").title()
        self.state = AgentState(role=role)
        self.message_history: List[AgentMessage] = []
        self.memory: List[Dict] = []  # Short-term memory
        self.client = None
        self.model = config.OPENROUTER_MODEL
        
        self._init_llm_client()
        
    def _init_llm_client(self):
        """Initialize OpenRouter client"""
        api_key = config.OPENROUTER_API_KEY
        if not api_key:
            print(f"⚠️ {self.name}: No API key, running in offline mode")
            return
            
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=config.OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://nifty-scalping-agent.app",
                    "X-Title": f"Trading Desk - {self.name}"
                }
            )
            print(f"✅ {self.name} initialized")
        except Exception as e:
            print(f"❌ {self.name} init error: {e}")
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Each agent must define its own system prompt"""
        pass
    
    @abstractmethod
    def process_request(self, request: Dict, context: Dict = None) -> Dict:
        """Process a request and return response"""
        pass
    
    def think(self, prompt: str, context: Dict = None) -> str:
        """Use LLM to reason about a problem"""
        if not self.client:
            return self._offline_think(prompt, context)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add context from memory
        if self.memory:
            memory_context = "\n".join([
                f"Previous insight: {m.get('insight', '')}" 
                for m in self.memory[-5:]
            ])
            messages.append({
                "role": "system", 
                "content": f"Recent context:\n{memory_context}"
            })
        
        # Add the current prompt
        if context:
            prompt = f"Context:\n{json.dumps(context, indent=2)}\n\nTask: {prompt}"
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content
            
            # Store in memory
            self.memory.append({
                "prompt": prompt[:200],
                "insight": result[:500],
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep memory bounded
            if len(self.memory) > 20:
                self.memory = self.memory[-20:]
            
            return result
            
        except Exception as e:
            print(f"{self.name} thinking error: {e}")
            return self._offline_think(prompt, context)
    
    def _offline_think(self, prompt: str, context: Dict = None) -> str:
        """Fallback reasoning when LLM is unavailable"""
        return f"[{self.name}] Analysis based on available data. (Offline mode)"
    
    def send_message(self, receiver: AgentRole, message_type: MessageType, 
                     content: Dict, priority: int = 2) -> AgentMessage:
        """Create a message to send to another agent"""
        message = AgentMessage(
            sender=self.role,
            receiver=receiver,
            message_type=message_type,
            content=content,
            priority=priority,
            conversation_id=f"{self.role.value}_{datetime.now().strftime('%H%M%S')}"
        )
        self.message_history.append(message)
        self.state.messages_processed += 1
        return message
    
    def receive_message(self, message: AgentMessage) -> Dict:
        """Process an incoming message"""
        self.message_history.append(message)
        self.state.status = "processing"
        self.state.current_task = f"Processing {message.message_type.value}"
        
        response = self._handle_message(message)
        
        self.state.status = "idle"
        self.state.current_task = None
        self.state.last_activity = datetime.now().isoformat()
        
        return response
    
    def _handle_message(self, message: AgentMessage) -> Dict:
        """Handle different message types"""
        handlers = {
            MessageType.ANALYSIS_REQUEST: self._handle_analysis_request,
            MessageType.STRATEGY_PROPOSAL: self._handle_strategy_proposal,
            MessageType.RISK_ASSESSMENT: self._handle_risk_assessment,
            MessageType.APPROVAL_REQUEST: self._handle_approval_request,
            MessageType.MARKET_UPDATE: self._handle_market_update,
            MessageType.CHAT: self._handle_chat,
        }
        
        handler = handlers.get(message.message_type, self._handle_default)
        return handler(message)
    
    def _handle_analysis_request(self, message: AgentMessage) -> Dict:
        return {"status": "received", "handler": "analysis"}
    
    def _handle_strategy_proposal(self, message: AgentMessage) -> Dict:
        return {"status": "received", "handler": "strategy"}
    
    def _handle_risk_assessment(self, message: AgentMessage) -> Dict:
        return {"status": "received", "handler": "risk"}
    
    def _handle_approval_request(self, message: AgentMessage) -> Dict:
        return {"status": "received", "handler": "approval"}
    
    def _handle_market_update(self, message: AgentMessage) -> Dict:
        return {"status": "received", "handler": "market_update"}
    
    def _handle_chat(self, message: AgentMessage) -> Dict:
        response = self.think(message.content.get("message", ""), message.content)
        return {"response": response}
    
    def _handle_default(self, message: AgentMessage) -> Dict:
        return {"status": "unhandled", "message_type": message.message_type.value}
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            "name": self.name,
            "role": self.role.value,
            "status": self.state.status,
            "current_task": self.state.current_task,
            "last_activity": self.state.last_activity,
            "messages_processed": self.state.messages_processed,
            "insights_generated": self.state.insights_generated,
            "memory_size": len(self.memory)
        }
    
    def clear_memory(self):
        """Clear agent's short-term memory"""
        self.memory = []
        
    def __repr__(self):
        return f"<{self.name} ({self.role.value}) - {self.state.status}>"
