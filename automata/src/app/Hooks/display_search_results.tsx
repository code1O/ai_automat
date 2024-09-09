import React, { useState } from "react";

type needProps = {
    connectInput: () => void;
    connectView: () => void;
}

export function display({connectInput} :needProps){
    const [ key, setKey ] = useState<string | null>(null);
    const [ componentShow, setComponentShow ] = useState<React.ReactNode | null>(null)
    const keyUp = (event: any) => {
        setKey(event.key);
    }
    
}