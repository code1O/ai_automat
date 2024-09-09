
type inputProps =  {
    keyUpEvent?: () => void;
    onButtonClick?: () => void;
    textHolder: string;
}

export const SearchInput = ({keyUpEvent, onButtonClick, textHolder}: inputProps) => {
    return(
    <div className="w-full max-w-sm min-w-[480px] relative mt-16">
    <div className="relative w-full mt-16">
        <input
        className="w-full h-10 pr-11 pl-3 py-2 bg-transparent rounded-xl placeholder:text-slate-400 border border-slate-300 focus:border-slate-400 hover:border-slate-300 shadow-lg focus:shadow-md duration-200 ease transition focus:outline-none"
        placeholder={textHolder}
        onKeyUp={keyUpEvent}
        />
        <button
        className="absolute h-8 w-8 right-1 top-1 my-auto px-2 flex items-center bg-slate-800 rounded-lg hover:bg-slate-700"
        type="button"
        onClick={onButtonClick}
        >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="#FFF" className="w-8 h-8">
            <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
        </svg>
        </button>
    </div>
    </div>
    )
}

export const ChatQueryInput = ({onButtonClick, textHolder}: inputProps) => {
    return (
        <div>
            <div>
                <input placeholder={textHolder} className=""/>
                <button onClick={onButtonClick} className=""></button>
            </div>
        </div>
    )
}

const SomeInput = (props: any) =>{
    return (
        <div className="relative w-full mt-4">
            <input id="UsernameInput" className="w-[400px] rounded-md pr-11 bg-transparent border border-slate-100 placeholder:text-slate-300 duration-200 ease transition hover:opacity-50 outline outline-width-2 outline-1 outline-transparent ml-2" placeholder={props.placeholder} type={props.typeInput}></input>
        </div>
    )
}
export const AccountInputs = ({onButtonClick}: inputProps) => {
    return (
        <div className="max-w-[400px] relative mt-4">
            <SomeInput placeholder="Email..." typeInput="email"/>
            <SomeInput placeholder="Phone number..." typeInput="tel"/>
            <SomeInput placeholder="Username..." type="text"/>
            <SomeInput placeholder="Password..." typeInput="password"/>
            <div className="flex flex-row justify-evenly ml-[-65px] mt-6">

                <button className="right-1 top-1 my-auto px-2 flex  bg-slate-800 rounded-lg hover:bg-slate-700" type="button" onClick={onButtonClick}>Save data</button>

                <button className="right-1 top-1 my-auto px-2 flex  bg-slate-800 rounded-lg hover:bg-slate-700" type="button" onClick={onButtonClick}>Continue anonymous</button>
            </div>
        </div>
    )
}