import { cn } from '@/lib/utils';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Bot, FileText, User, ChevronDown } from 'lucide-react';
import { Skeleton } from './ui/skeleton';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';
import { Button } from './ui/button';
import { GolemStats } from './golem-stats';

export type Message = {
  role: 'user' | 'assistant';
  content: string;
  file?: {
    name: string;
  };
  golemStats?: any;
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
          'max-w-2xl rounded-lg p-3 text-sm shadow-md',
          isUser
            ? 'rounded-br-none bg-primary text-primary-foreground'
            : 'rounded-bl-none bg-card text-card-foreground'
        )}
      >
        {message.file && (
          <div className="mb-2 flex items-center gap-2 rounded-md border border-primary-foreground/20 bg-primary-foreground/10 p-2">
            <FileText className="h-4 w-4 shrink-0" />
            <span className="truncate font-medium">{message.file.name}</span>
          </div>
        )}
        {message.content && (
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        )}
        {message.role === 'assistant' && message.golemStats && (
          <div className="mt-2 text-xs text-card-foreground/80">
            <Collapsible>
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start gap-2 text-card-foreground/80 hover:text-card-foreground"
                >
                  <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
                  Show Golem Stats
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <GolemStats stats={message.golemStats} />
              </CollapsibleContent>
            </Collapsible>
          </div>
        )}
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
    <div className="flex items-start justify-start gap-3">
      <Avatar className="h-8 w-8">
        <AvatarFallback>
          <Bot className="h-5 w-5 text-primary" />
        </AvatarFallback>
      </Avatar>
      <div className="max-w-md rounded-lg rounded-bl-none bg-card p-3 text-card-foreground shadow-md">
        <div className="flex items-center gap-2">
          <Skeleton className="h-2 w-2 animate-bounce rounded-full [animation-delay:-0.3s]" />
          <Skeleton className="h-2 w-2 animate-bounce rounded-full [animation-delay:-0.15s]" />
          <Skeleton className="h-2 w-2 animate-bounce rounded-full" />
        </div>
      </div>
    </div>
  );
}
