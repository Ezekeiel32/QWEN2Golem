import { cn } from '@/lib/utils';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Bot, User } from 'lucide-react';
import { Skeleton } from './ui/skeleton';

export type Message = {
  role: 'user' | 'assistant';
  content: string;
};

type ChatMessageProps = {
  message: Message;
};

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  return (
    <div
      className={cn(
        'flex items-start gap-3',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <Avatar className="h-8 w-8">
          <AvatarFallback>
            <Bot className="h-5 w-5 text-primary" />
          </AvatarFallback>
        </Avatar>
      )}
      <div
        className={cn(
          'max-w-md rounded-lg p-3 text-sm shadow-md',
          isUser
            ? 'rounded-br-none bg-primary text-primary-foreground'
            : 'rounded-bl-none bg-card text-card-foreground'
        )}
      >
        <p className="leading-relaxed">{message.content}</p>
      </div>
      {isUser && (
        <Avatar className="h-8 w-8">
          <AvatarFallback>
            <User className="h-5 w-5 text-primary" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}

export function LoadingMessage() {
  return (
    <div className="flex items-start gap-3 justify-start">
      <Avatar className="h-8 w-8">
        <AvatarFallback>
          <Bot className="h-5 w-5 text-primary" />
        </AvatarFallback>
      </Avatar>
      <div className="max-w-md rounded-lg p-3 shadow-md rounded-bl-none bg-card text-card-foreground">
        <div className="flex items-center gap-2">
            <Skeleton className="h-2 w-2 rounded-full animate-bounce [animation-delay:-0.3s]" />
            <Skeleton className="h-2 w-2 rounded-full animate-bounce [animation-delay:-0.15s]" />
            <Skeleton className="h-2 w-2 rounded-full animate-bounce" />
        </div>
      </div>
    </div>
  )
}
